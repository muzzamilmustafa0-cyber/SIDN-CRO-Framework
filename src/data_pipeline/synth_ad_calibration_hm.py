"""
SIDN-CRO data pipeline -- Cross-replication (Stage 2 / H&M).

Calibrated cooperative advertising spend for the H&M weekly panel.  The
modelling structure matches the DataCo / Olist branches; constants are
calibrated to fast-fashion benchmarks for the 2018 - 2020 period:

* Product-level ad-to-sales ratios (H&M Annual Report 2019; Statista
  fast-fashion ad-spend 2019-2020):
        Garment Upper body  0.10  (high-volume, high-rotation)
        Garment Lower body  0.09
        Garment Full body   0.10
        Accessories         0.07
        Shoes               0.08
* Cooperative split  0.25 / 0.35 / 0.40
* Retailer (age-cohort) bias:
        Young   1.20  (digital-first, social ads heavy)
        Mid     1.05
        Mature  0.85
* Yearly seasonality with peaks at week 14 (Easter/Spring), week 35
  (Back-to-school), week 48 (BFCM).
* Log-normal multiplicative noise sigma=0.20 on the cooperative split.

Random seed 42.
"""

from __future__ import annotations
import logging
import sys
import numpy as np
import pandas as pd

from src.utils.paths import HM_INTERMEDIATE_DIR

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("synth_ad_hm")

SEED = 42

PRODUCT_AD_RATIO = {
    "Garment Upper body": 0.10,
    "Garment Lower body": 0.09,
    "Garment Full body":  0.10,
    "Accessories":        0.07,
    "Shoes":              0.08,
}
AD_TO_SALES_SD = 0.025
COOP_SPLIT     = {"b_s": 0.25, "b_m": 0.35, "b_r": 0.40}
RETAILER_LEVEL = {"Young": 1.20, "Mid": 1.05, "Mature": 0.85}
PEAKS = [(14, 0.20), (35, 0.10), (48, 0.30)]


def seasonality(weekofyear: np.ndarray) -> np.ndarray:
    s = np.ones_like(weekofyear, dtype=float)
    for w_peak, amp in PEAKS:
        s += amp * np.exp(-0.5 * ((weekofyear - w_peak) / 4.0) ** 2)
    return s


def main() -> None:
    rng   = np.random.default_rng(SEED)
    panel = pd.read_parquet(HM_INTERMEDIATE_DIR / "hm_weekly.parquet")
    log.info("Input shape %s", panel.shape)

    panel["ratio_mean"]      = panel["product"].map(PRODUCT_AD_RATIO)
    panel["seasonality"]     = seasonality(panel["weekofyear"].values)
    panel["retailer_factor"] = panel["retailer"].map(RETAILER_LEVEL)

    sales_floor = max(1.0, panel["sales"].quantile(0.10))
    sales_eff   = np.maximum(panel["sales"].values, sales_floor)
    eta   = rng.normal(0.0, AD_TO_SALES_SD, size=len(panel))
    ratio = np.clip(panel["ratio_mean"].values + eta, 0.02, 0.18)

    total_ad = (sales_eff
                * ratio
                * panel["seasonality"].values
                * panel["retailer_factor"].values)

    raw_shares = np.zeros((len(panel), 3))
    for i, (k, v) in enumerate(COOP_SPLIT.items()):
        raw_shares[:, i] = v * rng.lognormal(mean=0.0, sigma=0.20, size=len(panel))
    raw_shares = raw_shares / raw_shares.sum(axis=1, keepdims=True)

    panel["b_s"]     = total_ad * raw_shares[:, 0]
    panel["b_m"]     = total_ad * raw_shares[:, 1]
    panel["b_r"]     = total_ad * raw_shares[:, 2]
    panel["b_total"] = total_ad
    panel = panel.drop(columns=["seasonality", "retailer_factor", "ratio_mean"])

    out = HM_INTERMEDIATE_DIR / "hm_weekly_ad.parquet"
    panel.to_parquet(out, index=False)
    log.info("Wrote %s shape=%s", out, panel.shape)

    log.info("Calibration check (USD/week, ratio=ad/sales):")
    for p in panel["product"].unique():
        sub = panel[panel["product"] == p]
        log.info("  %-22s b_s=$%5.0f b_m=$%5.0f b_r=$%6.0f total=$%7.0f"
                 "  ratio=%.1f%%",
                 p, sub["b_s"].mean(), sub["b_m"].mean(), sub["b_r"].mean(),
                 sub["b_total"].mean(),
                 100 * sub["b_total"].mean() / max(1.0, sub["sales"].mean()))


if __name__ == "__main__":
    sys.exit(main())
