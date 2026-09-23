"""
SIDN-CRO data pipeline -- Stage 2: Calibrated cooperative advertising spend
for the multi-product panel.

The DataCo dataset does not report advertising expenditure.  Following the
companion mathematical paper, the three-echelon cooperative-advertising
structure assigns three decision variables per (product i, retailer j, week t):

    b_{s,i,j,t}     supplier ad spend                   ($)
    b_{m,i,j,t}     manufacturer ad spend               ($)
    b_{r,i,j,t}     retailer ad spend                   ($)

Calibration
-----------
* Total ad-to-sales ratio centred on category-specific industry benchmarks
  (US Census Annual Retail Trade, Statista 2023, Deloitte Global Powers of
  Retailing):
        Apparel / fashion  : 8 %  (range 4 - 12 %)
        Sports merchandise : 5 %  (Fan Shop / Golf / Outdoors)
        Footwear           : 7 %
* Cooperative split (Bergen-John 2008; Aust-Buscher 2014):
        supplier = 0.25, manufacturer = 0.35, retailer = 0.40.
* Yearly seasonality with peaks before Easter (week 14), back-to-school
  (week 35) and Black-Friday / Christmas (week 48).
* Retailer-segment baseline shifts: Corporate spend higher, Home Office lower.
* Log-normal multiplicative noise (sigma=0.20) on the cooperative split.

Random seed 42 is fixed across the entire generation.
"""

from __future__ import annotations
import logging
import sys
import numpy as np
import pandas as pd

from src.utils.paths import INTERMEDIATE_DIR

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("synth_ad")

SEED = 42

PRODUCT_AD_RATIO = {                  # mean ad-to-sales ratio per category
    "Fan Shop": 0.05,
    "Apparel":  0.08,
    "Golf":     0.05,
    "Footwear": 0.07,
    "Outdoors": 0.05,
}
AD_TO_SALES_SD = 0.025

COOP_SPLIT  = {"b_s": 0.25, "b_m": 0.35, "b_r": 0.40}

RETAILER_LEVEL = {
    "Consumer":    1.05,
    "Corporate":   1.20,
    "Home Office": 0.90,
}

PEAKS = [(14, 0.20),
         (35, 0.10),
         (48, 0.30)]


def seasonality(weekofyear: np.ndarray) -> np.ndarray:
    s = np.ones_like(weekofyear, dtype=float)
    for w_peak, amp in PEAKS:
        s += amp * np.exp(-0.5 * ((weekofyear - w_peak) / 4.0) ** 2)
    return s


def main() -> None:
    rng   = np.random.default_rng(SEED)
    panel = pd.read_parquet(INTERMEDIATE_DIR / "dataco_weekly.parquet")
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

    out = INTERMEDIATE_DIR / "dataco_weekly_ad.parquet"
    panel.to_parquet(out, index=False)
    log.info("Wrote %s shape=%s", out, panel.shape)

    log.info("Calibration check (per product, USD/week, ratio=ad/sales):")
    for p in panel["product"].unique():
        sub = panel[panel["product"] == p]
        log.info("  %-9s b_s=$%5.0f  b_m=$%5.0f  b_r=$%5.0f  total=$%6.0f"
                 "  ratio=%.1f%%",
                 p, sub["b_s"].mean(), sub["b_m"].mean(), sub["b_r"].mean(),
                 sub["b_total"].mean(),
                 100 * sub["b_total"].mean() / max(1.0, sub["sales"].mean()))


if __name__ == "__main__":
    sys.exit(main())
