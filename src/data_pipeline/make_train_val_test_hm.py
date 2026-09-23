"""
SIDN-CRO data pipeline -- Cross-replication (Stage 4 / H&M).

Train / Val / Test split for the H&M panel.  Same augmentation parameters
as DataCo / Olist (k1=k2=0.030, k3=0.040, alpha_inc=0.80, eps_sd=0.05).

Splitting (chronological)
    2018-09-17 ... 2020-02-29  Train       (~73 %)
    2020-03-01 ... 2020-05-31  Validation  (~13 %)
    2020-06-01 ... 2020-09-21  Test        (~14 %)

The 2020-Q1 boundary deliberately straddles the early-COVID disruption
window, providing a real distribution-shift challenge for the SIDN-CRO
robustness experiments.
"""

from __future__ import annotations
import logging
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.utils.paths import (HM_INTERMEDIATE_DIR, HM_TRAIN_DIR, HM_VAL_DIR,
                             HM_TEST_DIR, HM_PREPROCESSED_DIR)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("make_split_hm")

SEED        = 44
K1, K2, K3  = 0.030, 0.030, 0.040
ALPHA_INC   = 0.80
NOISE_SD    = 0.05

PANEL_START = pd.Timestamp("2018-09-17")
PANEL_END   = pd.Timestamp("2020-09-21")
TRAIN_END   = pd.Timestamp("2020-03-01")
VAL_END     = pd.Timestamp("2020-06-01")

CONT_FEATS = ["p", "online_share",
              "b_s", "b_m", "b_r",
              "u_s", "u_m", "u_r",
              "n_unique_skus",
              "sin_woy", "cos_woy", "year_norm"]

PRODUCT_PREFIX  = "product_"
RETAILER_PREFIX = "retailer_"
TARGET_COL      = "qty_aug"


def trim_window(panel):
    n0 = len(panel)
    panel = panel[(panel["week_start"] >= PANEL_START) &
                  (panel["week_start"] <  PANEL_END)].copy()
    log.info("Trimmed: %d -> %d (-%d)", n0, len(panel), n0 - len(panel))
    return panel


def add_engineered(panel):
    panel["sin_woy"]   = np.sin(2 * np.pi * panel["weekofyear"] / 52.0)
    panel["cos_woy"]   = np.cos(2 * np.pi * panel["weekofyear"] / 52.0)
    panel["year_norm"] = (panel["year"] - 2018) / 2.0
    panel = panel.rename(columns={"qty": "qty_obs"})
    return panel


def apply_augmentation(panel, rng):
    panel["ad_eff"]  = (1.0
                       + K1 * np.log1p(panel["b_s"])
                       + K2 * np.log1p(panel["b_m"])
                       + K3 * np.log1p(panel["b_r"]))
    panel["cir_eff"] = 1.0 + ALPHA_INC * panel["avg_circ"]
    eps = rng.normal(0.0, NOISE_SD, size=len(panel))
    panel["qty_baseline"] = panel["qty_obs"]
    panel["qty_aug"] = panel["qty_obs"] * panel["ad_eff"] * panel["cir_eff"] * np.exp(eps)
    return panel


def add_one_hots(panel):
    prod_dummies = pd.get_dummies(panel["product"].str.replace(" ", "_"),
                                  prefix="product").astype(int)
    retl_dummies = pd.get_dummies(panel["retailer"].str.replace(" ", "_"),
                                  prefix="retailer").astype(int)
    return pd.concat([panel, prod_dummies, retl_dummies], axis=1)


def split_panel(panel):
    train = panel[panel["week_start"] <  TRAIN_END]
    val   = panel[(panel["week_start"] >= TRAIN_END) & (panel["week_start"] < VAL_END)]
    test  = panel[panel["week_start"] >= VAL_END]
    return train, val, test


def fit_scalers(train):
    return (StandardScaler().fit(train[CONT_FEATS].values),
            StandardScaler().fit(train[[TARGET_COL]].values))


def cat_columns(df):
    return sorted([c for c in df.columns
                   if (c.startswith(PRODUCT_PREFIX) or c.startswith(RETAILER_PREFIX))
                   and c not in {"product_id", "retailer_id"}])


def write_split(name, df, out_dir, x_scaler, y_scaler):
    cat_cols = cat_columns(df)
    cont    = x_scaler.transform(df[CONT_FEATS].values).astype("float32")
    target  = y_scaler.transform(df[[TARGET_COL]].values).astype("float32").ravel()
    cat     = df[cat_cols].values.astype("float32")
    np.save(out_dir / f"{name}_X_cont.npy", cont)
    np.save(out_dir / f"{name}_X_cat.npy",  cat)
    np.save(out_dir / f"{name}_y.npy",      target)
    df.reset_index(drop=True).to_parquet(out_dir / f"{name}_full.parquet", index=False)
    log.info("  %-5s n=%4d  cont_dim=%d  cat_dim=%d  active=%d",
             name, len(df), cont.shape[1], cat.shape[1], int(df["had_orders"].sum()))


def main():
    rng   = np.random.default_rng(SEED)
    panel = pd.read_parquet(HM_INTERMEDIATE_DIR / "hm_weekly_full.parquet")
    log.info("Input shape %s", panel.shape)

    panel = trim_window(panel)
    panel = add_engineered(panel)
    panel = apply_augmentation(panel, rng)
    panel = add_one_hots(panel)

    train, val, test = split_panel(panel)
    log.info("Split sizes  train=%d  val=%d  test=%d", len(train), len(val), len(test))

    x_scaler, y_scaler = fit_scalers(train)
    joblib.dump(x_scaler, HM_PREPROCESSED_DIR / "x_scaler.joblib")
    joblib.dump(y_scaler, HM_PREPROCESSED_DIR / "y_scaler.joblib")

    write_split("train", train, HM_TRAIN_DIR, x_scaler, y_scaler)
    write_split("val",   val,   HM_VAL_DIR,   x_scaler, y_scaler)
    write_split("test",  test,  HM_TEST_DIR,  x_scaler, y_scaler)

    schema = {
        "cont_feats": CONT_FEATS,
        "cat_feats" : cat_columns(panel),
        "target"    : TARGET_COL,
        "products"  : sorted(panel["product"].unique().tolist()),
        "retailers" : sorted(panel["retailer"].unique().tolist()),
        "augmentation_params": {"k1": K1, "k2": K2, "k3": K3,
                                "alpha_inc": ALPHA_INC, "noise_sd": NOISE_SD,
                                "seed": SEED},
        "split": {"panel_start": str(PANEL_START.date()),
                  "panel_end":   str(PANEL_END.date()),
                  "train_end":   str(TRAIN_END.date()),
                  "val_end":     str(VAL_END.date())},
    }
    with open(HM_PREPROCESSED_DIR / "schema.json", "w") as f:
        json.dump(schema, f, indent=2)
    log.info("Wrote schema.json and scalers.")


if __name__ == "__main__":
    sys.exit(main())
