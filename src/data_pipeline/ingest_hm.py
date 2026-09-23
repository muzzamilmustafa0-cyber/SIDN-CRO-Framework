"""
SIDN-CRO data pipeline -- Cross-replication (Stage 1 / H&M Personalised Fashion).

H&M Personalised Fashion Recommendations (Kaggle 2022; HuggingFace mirror
einrafh/hnm-fashion-recommendations-data; CC BY 4.0).  This panel adds a
real apparel-specific transactional source covering Sep 2018 - Sep 2020
(31.8 M rows) so the SIDN-CRO benchmark spans both classic OR datasets and
real fashion-retail transactions.

Mapping
-------
* Product i  in {Garment Upper body, Garment Lower body, Garment Full body,
                  Accessories, Shoes}   -- top-5 product_group_name (~92 %
                                            coverage)

* Retailer j in {Young, Mid, Mature}    -- age cohorts of the 1.37 M H&M
                                            customers, defined by the
                                            cumulative age distribution:
                                                Young  : age <= 25  (~28 %)
                                                Mid    : 26 - 45     (~46 %)
                                                Mature : >= 46       (~26 %)
                                            These cohorts have distinct price
                                            elasticities, advertising
                                            responsiveness, and circular-
                                            preference profiles.

* Calendar    weekly (ISO week)

Columns produced
----------------
For every (product i, retailer j, week t):

    p_ijt          mean unit price (price column already deflated to [0,1] by
                   the HuggingFace mirror; we rescale to USD using H&M's
                   publicly disclosed ASP of ~25 USD)
    qty_ijt        line count (each row = one article)
    sales_ijt      sum of price (in re-scaled USD)
    n_orders_ijt   distinct customer-day orders
    sales_channel  share of online (channel 2) vs in-store (channel 1)
    n_unique_skus  distinct articles
"""

from __future__ import annotations
import logging
import sys
import numpy as np
import pandas as pd

from src.utils.paths import (HM_ARTICLES, HM_CUSTOMERS, HM_TRANSACTIONS,
                             HM_INTERMEDIATE_DIR)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("ingest_hm")

PRODUCTS = ["Garment Upper body", "Garment Lower body", "Garment Full body",
            "Accessories", "Shoes"]

# H&M discloses an average sell price of ~25 USD across product mix
# (annual reports 2019, 2020).  The HuggingFace mirror normalises price to
# [0, 1] by dividing by the per-product max; we rescale uniformly so the
# mean per-line price is ~25 USD.
PRICE_RESCALE_TO_USD = 250.0


def cohort_label(age) -> str:
    if pd.isna(age):
        return "Mid"   # default bucket for missing-age customers
    if age <= 25:
        return "Young"
    if age <= 45:
        return "Mid"
    return "Mature"


def load_articles_customers():
    log.info("Loading articles + customers...")
    art = pd.read_parquet(HM_ARTICLES,
                          columns=["article_id", "product_group_name"])
    art = art[art["product_group_name"].isin(PRODUCTS)]
    art = art.rename(columns={"product_group_name": "product"})
    log.info("  articles in scope: %d", len(art))

    cust = pd.read_parquet(HM_CUSTOMERS, columns=["customer_id", "age"])
    cust["retailer"] = cust["age"].apply(cohort_label)
    log.info("  customers: %d  cohort split: %s",
             len(cust),
             cust["retailer"].value_counts().to_dict())
    return art, cust


def load_and_aggregate_transactions(art: pd.DataFrame,
                                    cust: pd.DataFrame) -> pd.DataFrame:
    """Stream-read transactions in chunks via pyarrow for memory efficiency."""
    import pyarrow.parquet as pq

    log.info("Streaming transactions...")
    pf = pq.ParquetFile(HM_TRANSACTIONS)

    art_lookup  = art.set_index("article_id")["product"]
    cust_lookup = cust.set_index("customer_id")["retailer"]

    weekly = []
    n_rows = 0
    for batch in pf.iter_batches(batch_size=2_000_000,
                                 columns=["t_dat", "customer_id",
                                          "article_id", "price",
                                          "sales_channel_id"]):
        df = batch.to_pandas()
        n_rows += len(df)
        df["product"]  = df["article_id"].map(art_lookup)
        df = df.dropna(subset=["product"])
        df["retailer"] = df["customer_id"].map(cust_lookup)
        df["retailer"] = df["retailer"].fillna("Mid")
        df["t_dat"]    = pd.to_datetime(df["t_dat"], errors="coerce")
        df = df.dropna(subset=["t_dat"])
        df["week_start"] = df["t_dat"].dt.to_period("W").dt.start_time
        df["price_usd"]  = df["price"] * PRICE_RESCALE_TO_USD
        df["online"]     = (df["sales_channel_id"] == 2).astype(int)

        agg = (df.groupby(["product", "retailer", "week_start"])
                 .agg(p           =("price_usd", "mean"),
                      qty         =("price_usd", "size"),
                      sales       =("price_usd", "sum"),
                      n_orders    =("customer_id", "nunique"),
                      online_share=("online", "mean"),
                      n_unique_skus=("article_id", "nunique"))
                 .reset_index())
        weekly.append(agg)
        log.info("  processed %d transactions so far (cumulative agg=%d rows)",
                 n_rows, sum(len(w) for w in weekly))

    panel = (pd.concat(weekly, ignore_index=True)
               .groupby(["product", "retailer", "week_start"])
               .agg(p          =("p",            "mean"),
                    qty        =("qty",          "sum"),
                    sales      =("sales",        "sum"),
                    n_orders   =("n_orders",     "sum"),
                    online_share=("online_share","mean"),
                    n_unique_skus=("n_unique_skus","sum"))
               .reset_index())

    panel["product_id"]  = panel["product"].map(
        {p: i + 1 for i, p in enumerate(PRODUCTS)})
    panel["retailer_id"] = panel["retailer"].map(
        {r: i + 1 for i, r in enumerate(["Young", "Mid", "Mature"])})
    panel["year"]       = panel["week_start"].dt.year
    panel["month"]      = panel["week_start"].dt.month
    panel["weekofyear"] = panel["week_start"].dt.isocalendar().week.astype(int)
    panel = panel.sort_values(["product_id", "retailer_id", "week_start"]).reset_index(drop=True)
    return panel


def reindex_complete_grid(panel: pd.DataFrame) -> pd.DataFrame:
    weeks = pd.date_range(panel["week_start"].min(),
                          panel["week_start"].max(), freq="W-MON")
    keys = panel[["product", "product_id", "retailer", "retailer_id"]].drop_duplicates()
    grid = keys.merge(pd.DataFrame({"week_start": weeks}), how="cross")
    full = grid.merge(panel,
                      on=["product", "product_id", "retailer", "retailer_id", "week_start"],
                      how="left")
    full["had_orders"] = (~full["qty"].isna()).astype(int)
    full[["qty", "sales", "n_orders", "n_unique_skus"]] = (
        full[["qty", "sales", "n_orders", "n_unique_skus"]].fillna(0))
    ffill_cols = ["p", "online_share"]
    full[ffill_cols] = (
        full.groupby(["product_id", "retailer_id"])[ffill_cols]
            .transform(lambda s: s.ffill().bfill())
    )
    full["year"]       = full["week_start"].dt.year
    full["month"]      = full["week_start"].dt.month
    full["weekofyear"] = full["week_start"].dt.isocalendar().week.astype(int)
    full = full.sort_values(["product_id", "retailer_id", "week_start"]).reset_index(drop=True)
    return full


def main() -> None:
    art, cust = load_articles_customers()
    panel     = load_and_aggregate_transactions(art, cust)
    panel     = reindex_complete_grid(panel)

    out = HM_INTERMEDIATE_DIR / "hm_weekly.parquet"
    panel.to_parquet(out, index=False)
    log.info("Wrote %s shape=%s", out, panel.shape)

    log.info("Per-product summary:")
    for p in PRODUCTS:
        sub = panel[panel["product"] == p]
        active = int(sub["had_orders"].sum())
        log.info("  %-22s n=%4d active=%4d (%.0f%%)  qty=%.1f  p=$%.2f  sales=$%.0f",
                 p, len(sub), active, 100 * active / max(1, len(sub)),
                 sub["qty"].mean(), sub["p"].mean(), sub["sales"].mean())

    log.info("Per-retailer summary:")
    for r in ["Young", "Mid", "Mature"]:
        sub = panel[panel["retailer"] == r]
        active = int(sub["had_orders"].sum())
        log.info("  %-7s n=%4d active=%4d (%.0f%%)  qty=%.1f  p=$%.2f",
                 r, len(sub), active, 100 * active / max(1, len(sub)),
                 sub["qty"].mean(), sub["p"].mean())


if __name__ == "__main__":
    sys.exit(main())
