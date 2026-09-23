"""
SIDN-CRO data pipeline -- Cross-replication (Stage 1 / Olist).

We replicate the DataCo data pipeline on the Brazilian Olist e-commerce
public dataset (Sionek 2018; CC BY-NC-SA 4.0; ~99 k orders, 2016-09 to
2018-10, 73 product categories, 27 Brazilian states) so that the SIDN-CRO
results can be cross-validated on a structurally different public dataset
in line with top-decile journal expectations (IJPE / IJPR / EJOR).

Mapping
-------
* Product i in {bed_bath_table, health_beauty, sports_leisure,
                furniture_decor, computers_accessories}     (top-5 cats by
                                                             order volume)
* Retailer j in {SP, Southeast, Other}                       (3 macro-regions:
                                                             Sao Paulo state,
                                                             Southeast minus SP,
                                                             rest of Brazil)
* Calendar     weekly (ISO week)

For every (product i, retailer j, week t) we compute:

    p_ijt          mean unit price (price column of order_items)
    freight_ijt    mean unit freight value
    qty_ijt        sum of items (line-count -- Olist quantities are 1 per row)
    sales_ijt      sum of price + freight
    n_orders_ijt   #distinct orders
    weight_ijt     mean product weight (g)

Records are filtered to delivered or invoiced orders and reindexed to a
complete (product x retailer x week) grid.
"""

from __future__ import annotations
import logging
import sys
import numpy as np
import pandas as pd

from src.utils.paths import (OLIST_ORDERS_CSV, OLIST_ITEMS_CSV,
                             OLIST_PRODUCTS_CSV, OLIST_CUSTOMERS_CSV,
                             OLIST_TRANSLATION_CSV, OLIST_INTERMEDIATE_DIR)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("ingest_olist")

PRODUCTS = ["bed_bath_table", "health_beauty", "sports_leisure",
            "furniture_decor", "computers_accessories"]

SP_STATES        = ["SP"]
SOUTHEAST_STATES = ["RJ", "MG", "ES"]      # Southeast minus SP
# everything else -> "Other"

VALID_STATUSES = ["delivered", "invoiced", "shipped", "approved"]


def load_raw():
    log.info("Loading Olist tables...")
    ords = pd.read_csv(OLIST_ORDERS_CSV)
    items = pd.read_csv(OLIST_ITEMS_CSV)
    prods = pd.read_csv(OLIST_PRODUCTS_CSV)
    cust  = pd.read_csv(OLIST_CUSTOMERS_CSV)
    trans = pd.read_csv(OLIST_TRANSLATION_CSV)
    log.info("  orders:%s items:%s prods:%s cust:%s trans:%s",
             ords.shape, items.shape, prods.shape, cust.shape, trans.shape)
    return ords, items, prods, cust, trans


def merge_and_filter(ords, items, prods, cust, trans):
    df = items.merge(ords[["order_id", "customer_id", "order_status",
                           "order_purchase_timestamp"]],
                     on="order_id", how="left")
    df = df.merge(prods[["product_id", "product_category_name",
                         "product_weight_g"]],
                  on="product_id", how="left")
    df = df.merge(trans, on="product_category_name", how="left")
    df = df.merge(cust[["customer_id", "customer_state"]],
                  on="customer_id", how="left")

    n0 = len(df)
    df = df[df["order_status"].isin(VALID_STATUSES)]
    log.info("After Order Status in %s: %d (-%d)", VALID_STATUSES, len(df),
             n0 - len(df))

    df = df[df["product_category_name_english"].isin(PRODUCTS)]
    log.info("After product category in top-5: %d", len(df))

    return df


def assign_retailer(state: str) -> str:
    if state in SP_STATES:
        return "SP"
    if state in SOUTHEAST_STATES:
        return "Southeast"
    return "Other"


def add_time(df: pd.DataFrame) -> pd.DataFrame:
    df["order_date"] = pd.to_datetime(df["order_purchase_timestamp"],
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
    df["retailer"] = df["customer_state"].map(assign_retailer)
    g = df.groupby(["product_category_name_english", "retailer", "week_start"])

    panel = g.agg(
        p             = ("price",         "mean"),
        freight       = ("freight_value", "mean"),
        qty           = ("order_id",      "count"),     # 1 row = 1 line
        sales         = ("price",         "sum"),
        freight_sum   = ("freight_value", "sum"),
        n_orders      = ("order_id",      "nunique"),
        weight        = ("product_weight_g", "mean"),
        year          = ("year",          "first"),
        month         = ("month",         "first"),
        weekofyear    = ("weekofyear",    "first"),
    ).reset_index()

    panel = panel.rename(columns={"product_category_name_english": "product"})
    panel["sales"] = panel["sales"] + panel["freight_sum"].fillna(0)
    panel = panel.drop(columns=["freight_sum"])
    panel["product_id"]  = panel["product"].map({p: i + 1 for i, p in enumerate(PRODUCTS)})
    retailer_order = ["SP", "Southeast", "Other"]
    panel["retailer_id"] = panel["retailer"].map({r: i + 1 for i, r in enumerate(retailer_order)})
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

    full[["qty", "sales", "n_orders"]] = full[["qty", "sales", "n_orders"]].fillna(0)

    ffill_cols = ["p", "freight", "weight"]
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
    ords, items, prods, cust, trans = load_raw()
    df = merge_and_filter(ords, items, prods, cust, trans)
    df = add_time(df)
    panel = aggregate_weekly(df)
    panel = reindex_complete_grid(panel)

    out = OLIST_INTERMEDIATE_DIR / "olist_weekly.parquet"
    panel.to_parquet(out, index=False)
    log.info("Wrote %s shape=%s", out, panel.shape)

    log.info("Per-product summary:")
    for p in PRODUCTS:
        sub = panel[panel["product"] == p]
        active = int(sub["had_orders"].sum())
        log.info("  %-25s n=%4d active=%4d (%.0f%%)  qty=%.1f  p=R$%.2f",
                 p, len(sub), active, 100 * active / max(1, len(sub)),
                 sub["qty"].mean(), sub["p"].mean())

    log.info("Per-retailer summary:")
    for r in ["SP", "Southeast", "Other"]:
        sub = panel[panel["retailer"] == r]
        active = int(sub["had_orders"].sum())
        log.info("  %-12s n=%4d active=%4d (%.0f%%)  qty=%.1f  p=R$%.2f",
                 r, len(sub), active, 100 * active / max(1, len(sub)),
                 sub["qty"].mean(), sub["p"].mean())


if __name__ == "__main__":
    sys.exit(main())
