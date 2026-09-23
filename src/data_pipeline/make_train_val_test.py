"""
SIDN-CRO data pipeline -- Stage 4: build features, augmented target and the
temporal Train / Validation / Test split for the multi-product panel.

Augmented demand signal
-----------------------
The DataCo `qty` series captures real seasonality, regional differences,
customer-segment mix and price patterns; we keep it as the baseline.  On top
of that, we apply the multiplicative advertising and circular uplifts of the
companion mathematical paper:

    Ad_eff_ijt  = 1 + k1 * ln(1 + b_s,ijt)
                    + k2 * ln(1 + b_m,ijt)
                    + k3 * ln(1 + b_r,ijt)            (Khorshidvand et al. eq. 9)

    Cir_eff_ijt = 1 + alpha_inc * avg_circ_ijt        (eq. 7)

    qty_aug_ijt = qty_ijt * Ad_eff_ijt * Cir_eff_ijt * exp(epsilon_ijt),
                  epsilon_ijt ~ N(0, 0.05^2)

Calibration of (k1, k2, k3, alpha_inc) follows the original paper:
                k1 = k2 = 0.030, k3 = 0.040, alpha_inc = 0.80.

Features written to disk
------------------------
Continuous : p, disc_rate, b_s, b_m, b_r, u_s, u_m, u_r,
             share_eu, share_latam, share_pac, share_usca, share_africa,
             sin_woy, cos_woy, year_norm
Categorical: 5 product one-hots + 3 retailer one-hots
Targets    : qty_aug              (regression target, ML)
             qty_obs              (raw DataCo demand, kept for diagnostics)
             qty_baseline         (qty_obs without ad/circular uplift)

Splitting (chronological, fully aligned across products and retailers)
----------------------------------------------------------------------
We trim the panel to the 2015-01-01 .. 2017-09-30 fully-active window
because beyond that point the DataCo tail drops to <30 % active cells
(an artefact of the simulated source rather than a real distribution
shift), then split:

    2015-01-01  ...  2016-12-31   ->  Train       (~73 %)
    2017-01-01  ...  2017-04-30   ->  Validation  (~12 %)
    2017-05-01  ...  2017-09-30   ->  Test        (~15 %)

The covariate shift from train to test is genuine (2017 prices and
seasonality differ from 2015-2016), and the entire panel is 100 % active,
giving ~2,100 observations for ML training and evaluation.
"""

from __future__ import annotations
import logging
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.utils.paths import (INTERMEDIATE_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR,
                             PREPROCESSED_DIR)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("make_split")

SEED        = 44
K1, K2, K3  = 0.030, 0.030, 0.040
ALPHA_INC   = 0.80
NOISE_SD    = 0.05

PANEL_START = pd.Timestamp("2015-01-01")
PANEL_END   = pd.Timestamp("2017-09-30")
TRAIN_END   = pd.Timestamp("2017-01-01")
VAL_END     = pd.Timestamp("2017-05-01")

CONT_FEATS = ["p", "disc_rate",
              "b_s", "b_m", "b_r",
              "u_s", "u_m", "u_r",
              "share_eu", "share_latam", "share_pac",
              "share_usca", "share_africa",
              "sin_woy", "cos_woy", "year_norm"]

PRODUCT_PREFIX  = "product_"
RETAILER_PREFIX = "retailer_"
TARGET_COL      = "qty_aug"


def add_engineered(panel: pd.DataFrame) -> pd.DataFrame:
    panel["sin_woy"]   = np.sin(2 * np.pi * panel["weekofyear"] / 52.0)
    panel["cos_woy"]   = np.cos(2 * np.pi * panel["weekofyear"] / 52.0)
    panel["year_norm"] = (panel["year"] - 2015) / 3.0
    panel = panel.rename(columns={"qty": "qty_obs"})
    return panel


def apply_augmentation(panel: pd.DataFrame, rng) -> pd.DataFrame:
    panel["ad_eff"]  = (1.0
                       + K1 * np.log1p(panel["b_s"])
                       + K2 * np.log1p(panel["b_m"])
                       + K3 * np.log1p(panel["b_r"]))
    panel["cir_eff"] = 1.0 + ALPHA_INC * panel["avg_circ"]

    eps = rng.normal(0.0, NOISE_SD, size=len(panel))
    panel["qty_baseline"] = panel["qty_obs"]
    panel["qty_aug"] = panel["qty_obs"] * panel["ad_eff"] * panel["cir_eff"] * np.exp(eps)
    return panel


def add_one_hots(panel: pd.DataFrame) -> pd.DataFrame:
    prod_dummies = pd.get_dummies(
        panel["product"].str.replace(" ", "_"), prefix="product"
    ).astype(int)
    retl_dummies = pd.get_dummies(
        panel["retailer"].str.replace(" ", "_"), prefix="retailer"
    ).astype(int)
    return pd.concat([panel, prod_dummies, retl_dummies], axis=1)


def trim_to_active_window(panel: pd.DataFrame) -> pd.DataFrame:
    n0 = len(panel)
    panel = panel[(panel["week_start"] >= PANEL_START) &
                  (panel["week_start"] <  PANEL_END)].copy()
    log.info("Trimmed inactive tail: %d -> %d rows (-%d)",
             n0, len(panel), n0 - len(panel))
    return panel


def split_panel(panel: pd.DataFrame):
    train = panel[panel["week_start"] <  TRAIN_END]
    val   = panel[(panel["week_start"] >= TRAIN_END) &
                  (panel["week_start"] <  VAL_END)]
    test  = panel[panel["week_start"] >= VAL_END]
    return train, val, test


def fit_scalers(train: pd.DataFrame):
    x_scaler = StandardScaler().fit(train[CONT_FEATS].values)
    y_scaler = StandardScaler().fit(train[[TARGET_COL]].values)
    return x_scaler, y_scaler


def cat_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns
            if (c.startswith(PRODUCT_PREFIX) or c.startswith(RETAILER_PREFIX))
            and c not in {"product_id", "retailer_id"}]
    return sorted(cols)


def write_split(name: str, df: pd.DataFrame, out_dir, x_scaler, y_scaler):
    cat_cols = cat_columns(df)
    cont    = x_scaler.transform(df[CONT_FEATS].values).astype("float32")
    target  = y_scaler.transform(df[[TARGET_COL]].values).astype("float32").ravel()
    cat     = df[cat_cols].values.astype("float32")

    np.save(out_dir / f"{name}_X_cont.npy", cont)
    np.save(out_dir / f"{name}_X_cat.npy",  cat)
    np.save(out_dir / f"{name}_y.npy",      target)
    df.reset_index(drop=True).to_parquet(out_dir / f"{name}_full.parquet",
                                         index=False)
    log.info("  %-5s n=%4d  cont_dim=%d  cat_dim=%d  active=%d",
             name, len(df), cont.shape[1], cat.shape[1],
             int(df["had_orders"].sum()))


def main() -> None:
    rng   = np.random.default_rng(SEED)
    panel = pd.read_parquet(INTERMEDIATE_DIR / "dataco_weekly_full.parquet")
    log.info("Input shape %s", panel.shape)

    panel = trim_to_active_window(panel)
    panel = add_engineered(panel)
    panel = apply_augmentation(panel, rng)
    panel = add_one_hots(panel)

    train, val, test = split_panel(panel)
    log.info("Split sizes  train=%d  val=%d  test=%d  (total=%d)",
             len(train), len(val), len(test), len(panel))

    x_scaler, y_scaler = fit_scalers(train)
    joblib.dump(x_scaler, PREPROCESSED_DIR / "x_scaler.joblib")
    joblib.dump(y_scaler, PREPROCESSED_DIR / "y_scaler.joblib")

    log.info("Writing splits:")
    write_split("train", train, TRAIN_DIR, x_scaler, y_scaler)
    write_split("val",   val,   VAL_DIR,   x_scaler, y_scaler)
    write_split("test",  test,  TEST_DIR,  x_scaler, y_scaler)

    schema = {
        "cont_feats": CONT_FEATS,
        "cat_feats" : cat_columns(panel),
        "target"    : TARGET_COL,
        "products"  : sorted(panel["product"].unique().tolist()),
        "retailers" : sorted(panel["retailer"].unique().tolist()),
        "augmentation_params": {
            "k1": K1, "k2": K2, "k3": K3,
            "alpha_inc": ALPHA_INC, "noise_sd": NOISE_SD,
            "seed": SEED,
        },
        "split": {
            "panel_start": str(PANEL_START.date()),
            "panel_end"  : str(PANEL_END.date()),
            "train_end"  : str(TRAIN_END.date()),
            "val_end"    : str(VAL_END.date()),
        },
    }
    with open(PREPROCESSED_DIR / "schema.json", "w") as f:
        json.dump(schema, f, indent=2)
    log.info("Wrote schema.json and scalers.")


if __name__ == "__main__":
    sys.exit(main())
