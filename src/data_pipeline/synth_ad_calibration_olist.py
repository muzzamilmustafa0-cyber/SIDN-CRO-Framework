"""
SIDN-CRO data pipeline -- Cross-replication (Stage 2 / Olist).

Generates calibrated cooperative advertising spend (b_s, b_m, b_r) for the
Olist multi-product weekly panel using the same modelling assumptions as the
DataCo branch but with Brazilian-context constants:

* Product-level ad-to-sales ratios calibrated to Brazilian online retail
  industry benchmarks (IAB Brasil 2018, Kantar Ibope 2017):
        bed_bath_table          0.05  (homewares)
        health_beauty           0.10  (cosmetics is high-ad)
        sports_leisure          0.07
        furniture_decor         0.05
        computers_accessories   0.06
* Cooperative split             0.25 / 0.35 / 0.40  (same as DataCo branch)
* Yearly seasonality            same triple-peak (Carnival ~ wk 8, Black-Friday
                                ~ wk 47, Christmas ~ wk 51 are dominant in
                                Brazilian retail; we re-use the
                                Easter / BTS / BFCM triple as a robust proxy).
* Retailer-region bias          SP=1.10  Southeast=0.95  Other=0.85
                                (advertising buys in Brazil concentrate in SP)
* Log-normal multiplicative noise on the cooperative split, sigma=0.20.

Random seed 42.
"""

from __future__ import annotations
import logging
import sys
import numpy as np
import pandas as pd

from src.utils.paths import OLIST_INTERMEDIATE_DIR

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("synth_ad_olist")

SEED = 42

PRODUCT_AD_RATIO = {
    "bed_bath_table":        0.05,
    "health_beauty":         0.10,
    "sports_leisure":        0.07,
    "furniture_decor":       0.05,
    "computers_accessories": 0.06,
}
AD_TO_SALES_SD = 0.025

COOP_SPLIT = {"b_s": 0.25, "b_m": 0.35, "b_r": 0.40}

RETAILER_LEVEL = {"SP": 1.10, "Southeast": 0.95, "Other": 0.85}

PEAKS = [(14, 0.20),    # Easter
         (35, 0.10),    # BTS
         (48, 0.30)]    # Black-Friday / Christmas


def seasonality(weekofyear: np.ndarray) -> np.ndarray:
    s = np.ones_like(weekofyear, dtype=float)
    for w_peak, amp in PEAKS:
        s += amp * np.exp(-0.5 * ((weekofyear - w_peak) / 4.0) ** 2)
    return s


def main() -> None:
    rng   = np.random.default_rng(SEED)
    panel = pd.read_parquet(OLIST_INTERMEDIATE_DIR / "olist_weekly.parquet")
    log.info("Input shape %s products=%s",
             panel.shape, sorted(panel["product"].unique()))

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

    out = OLIST_INTERMEDIATE_DIR / "olist_weekly_ad.parquet"
    panel.to_parquet(out, index=False)
    log.info("Wrote %s shape=%s", out, panel.shape)

    log.info("Calibration check (BRL/week, ratio=ad/sales):")
    for p in panel["product"].unique():
        sub = panel[panel["product"] == p]
        log.info("  %-25s b_s=R$%5.0f b_m=R$%5.0f b_r=R$%5.0f total=R$%6.0f"
                 "  ratio=%.1f%%",
                 p, sub["b_s"].mean(), sub["b_m"].mean(), sub["b_r"].mean(),
                 sub["b_total"].mean(),
                 100 * sub["b_total"].mean() / max(1.0, sub["sales"].mean()))


if __name__ == "__main__":
    sys.exit(main())
