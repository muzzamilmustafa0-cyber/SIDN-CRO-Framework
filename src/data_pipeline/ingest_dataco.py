"""
SIDN-CRO data pipeline -- Stage 1: Ingest the DataCo Smart Supply Chain dataset.

We map DataCo onto the multi-product extension of the three-echelon structure
of the companion mathematical paper:

* Department      in {Fan Shop, Apparel, Golf,         -> 5 products i=1..5
                      Footwear, Outdoors}                  (top-5 by volume,
                                                           covering ~96 % of
                                                           total order rows)

* Customer Segment in {Consumer, Corporate,            -> 3 retailers j=1..3
                       Home Office}                       (B2C / B2B channels
                                                           served by the same
                                                           supplier-manufacturer
                                                           pair)

* Calendar         weekly aggregation                  -> planning period

The combined product x retailer x week panel is rebalanced to a complete
grid:  ~5 products * 3 retailers * 156 weeks = ~2,340 observations, with the
overwhelming majority of cells active for ML training.

For every (product i, retailer j, week t) we compute:

    p_ijt          mean Order Item Product Price
    disc_ijt       mean Order Item Discount Rate
    qty_ijt        sum  Order Item Quantity              (raw demand signal)
    sales_ijt      sum  Sales
    profit_ijt     sum  Order Profit Per Order
    n_orders_ijt   #orders
    market_*       share of orders shipped to each Market region

References
----------
Constante, F.; Silva, F.; Pereira, A. "DataCo SMART SUPPLY CHAIN FOR BIG DATA
ANALYSIS". Mendeley Data, V3, 2019. doi:10.17632/8gx2fvg2k6.3
"""

from __future__ import annotations
import logging
import sys
import pandas as pd
import numpy as np

from src.utils.paths import DATACO_CSV, INTERMEDIATE_DIR

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("ingest_dataco")

PRODUCTS  = ["Fan Shop", "Apparel", "Golf", "Footwear", "Outdoors"]   # i=1..5
RETAILERS = ["Consumer", "Corporate", "Home Office"]                  # j=1..3
VALID_STATUSES = ["COMPLETE", "CLOSED", "PROCESSING", "PENDING", "PENDING_PAYMENT"]


def load_raw() -> pd.DataFrame:
    log.info("Loading %s", DATACO_CSV)
    df = pd.read_csv(DATACO_CSV, encoding="latin-1", low_memory=False)
    log.info("Raw shape: %s", df.shape)
    return df


def filter_scope(df: pd.DataFrame) -> pd.DataFrame:
    n0 = len(df)
    df = df[df["Department Name"].isin(PRODUCTS)].copy()
    log.info("After Department in %s: %d rows (-%d)",
             PRODUCTS, len(df), n0 - len(df))

    df = df[df["Customer Segment"].isin(RETAILERS)]
    log.info("After Customer Segment in %s: %d rows", RETAILERS, len(df))

    df = df[df["Order Status"].isin(VALID_STATUSES)]
    log.info("After Order Status filter: %d rows", len(df))
    return df


def add_time(df: pd.DataFrame) -> pd.DataFrame:
    df["order_date"] = pd.to_datetime(df["order date (DateOrders)"],
                                      format="%m/%d/%Y %H:%M",
                                      errors="coerce")
    df = df.dropna(subset=["order_date"])
    df["week_start"] = df["order_date"].dt.to_period("W").dt.start_time
    df["year"]       = df["order_date"].dt.year
    df["month"]      = df["order_date"].dt.month
    df["weekofyear"] = df["order_date"].dt.isocalendar().week.astype(int)
    log.info("Order date span: %s -> %s",
             df["order_date"].min().date(), df["order_date"].max().date())
    return df


def aggregate_weekly(df: pd.DataFrame) -> pd.DataFrame:
    df["mkt_europe"]  = (df["Market"] == "Europe").astype(int)
    df["mkt_latam"]   = (df["Market"] == "LATAM").astype(int)
    df["mkt_pacific"] = (df["Market"] == "Pacific Asia").astype(int)
    df["mkt_usca"]    = (df["Market"] == "USCA").astype(int)
    df["mkt_africa"]  = (df["Market"] == "Africa").astype(int)

    g = df.groupby(["Department Name", "Customer Segment", "week_start"])

    panel = g.agg(
        p             = ("Order Item Product Price", "mean"),
        disc_rate     = ("Order Item Discount Rate", "mean"),
        qty           = ("Order Item Quantity",      "sum"),
        sales         = ("Sales",                    "sum"),
        profit        = ("Order Profit Per Order",   "sum"),
        n_orders      = ("Order Id",                 "nunique"),
        share_eu      = ("mkt_europe",               "mean"),
        share_latam   = ("mkt_latam",                "mean"),
        share_pac     = ("mkt_pacific",              "mean"),
        share_usca    = ("mkt_usca",                 "mean"),
        share_africa  = ("mkt_africa",               "mean"),
        year          = ("year",                     "first"),
        month         = ("month",                    "first"),
        weekofyear    = ("weekofyear",               "first"),
    ).reset_index()

    panel = panel.rename(columns={"Department Name":  "product",
                                  "Customer Segment": "retailer"})
    panel["product_id"]  = panel["product"].map({p: i + 1 for i, p in enumerate(PRODUCTS)})
    panel["retailer_id"] = panel["retailer"].map({r: i + 1 for i, r in enumerate(RETAILERS)})
    panel = panel.sort_values(["product_id", "retailer_id", "week_start"]).reset_index(drop=True)
    return panel


def reindex_complete_grid(panel: pd.DataFrame) -> pd.DataFrame:
    """Build a complete (product x retailer x week) grid; missing cells get
    qty/sales/profit/n_orders=0 and price/discount/market shares
    forward-filled within each (product, retailer)."""
    weeks = pd.date_range(panel["week_start"].min(),
                          panel["week_start"].max(),
                          freq="W-MON")
    keys = panel[["product", "product_id", "retailer", "retailer_id"]].drop_duplicates()
    grid = keys.merge(pd.DataFrame({"week_start": weeks}), how="cross")

    full = grid.merge(panel,
                      on=["product", "product_id", "retailer", "retailer_id", "week_start"],
                      how="left")
    full["had_orders"] = (~full["qty"].isna()).astype(int)

    fill_zero = ["qty", "sales", "profit", "n_orders"]
    full[fill_zero] = full[fill_zero].fillna(0)

    ffill_cols = ["p", "disc_rate",
                  "share_eu", "share_latam", "share_pac",
                  "share_usca", "share_africa"]
    full[ffill_cols] = (
        full.groupby(["product_id", "retailer_id"])[ffill_cols]
            .transform(lambda s: s.ffill().bfill())
    )

    full["year"]       = full["week_start"].dt.year
    full["month"]      = full["week_start"].dt.month
    full["weekofyear"] = full["week_start"].dt.isocalendar().week.astype(int)

    full = full.sort_values(
        ["product_id", "retailer_id", "week_start"]
    ).reset_index(drop=True)
    return full


def main() -> None:
    df = load_raw()
    df = filter_scope(df)
    df = add_time(df)
    panel = aggregate_weekly(df)
    panel = reindex_complete_grid(panel)

    out = INTERMEDIATE_DIR / "dataco_weekly.parquet"
    panel.to_parquet(out, index=False)
    log.info("Wrote %s shape=%s", out, panel.shape)

    log.info("Per-product summary (averaged across retailers and weeks):")
    for p in PRODUCTS:
        sub = panel[panel["product"] == p]
        active = int(sub["had_orders"].sum())
        log.info("  %-9s n=%4d active=%4d (%.0f%%)  qty=%.1f  p=$%.2f  sales=$%.0f",
                 p, len(sub), active, 100 * active / max(1, len(sub)),
                 sub["qty"].mean(), sub["p"].mean(), sub["sales"].mean())

    log.info("Per-retailer summary (averaged across products and weeks):")
    for r in RETAILERS:
        sub = panel[panel["retailer"] == r]
        active = int(sub["had_orders"].sum())
        log.info("  %-12s n=%4d active=%4d (%.0f%%)  qty=%.1f  p=$%.2f",
                 r, len(sub), active, 100 * active / max(1, len(sub)),
                 sub["qty"].mean(), sub["p"].mean())


if __name__ == "__main__":
    sys.exit(main())
