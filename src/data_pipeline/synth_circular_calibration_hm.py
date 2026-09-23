"""
SIDN-CRO data pipeline -- Cross-replication (Stage 3 / H&M).

Calibrated circular-input fractions u_s, u_m, u_r for the H&M panel
(2018-2020).  H&M was an early circular-fashion adopter (Conscious Collection
launched 2010; Recycled Cotton launched 2018; Take-Back program in 4,000+
stores by 2020), so trends are slightly higher than the DataCo / Olist branches:

* Linear trend per (product, echelon) calibrated against EEA Textile
  Circularity Indicator and H&M Sustainability Reports 2018-2020:
        Garment Upper body    u_s 0.012 -> 0.038, u_m 0.022 -> 0.055, u_r 0.030 -> 0.072
        Garment Lower body    similar ranges (denim with recycled cotton)
        Garment Full body     similar
        Accessories           lower (less recycling-friendly mix)
        Shoes                 lowest (recycled-rubber programs nascent in 2018)
* Retailer-cohort bias:
        Young   1.20  (active sustainability preference, Gen Z)
        Mid     1.00  (millennials, mixed preference)
        Mature  0.85  (price-driven, less sustainability-led)
* AR(1) noise (phi=0.85, sigma=0.0035).

Random seed 43.
"""

from __future__ import annotations
import logging
import sys
import numpy as np
import pandas as pd

from src.utils.paths import HM_INTERMEDIATE_DIR

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("synth_circ_hm")

SEED = 43

TREND = {
    "Garment Upper body": {"u_s": (0.012, 0.038), "u_m": (0.022, 0.055), "u_r": (0.030, 0.072)},
    "Garment Lower body": {"u_s": (0.014, 0.040), "u_m": (0.024, 0.058), "u_r": (0.032, 0.075)},
    "Garment Full body":  {"u_s": (0.011, 0.035), "u_m": (0.020, 0.052), "u_r": (0.028, 0.068)},
    "Accessories":        {"u_s": (0.008, 0.025), "u_m": (0.014, 0.040), "u_r": (0.020, 0.055)},
    "Shoes":              {"u_s": (0.005, 0.018), "u_m": (0.010, 0.030), "u_r": (0.015, 0.045)},
}

RETAILER_BIAS = {"Young": 1.20, "Mid": 1.00, "Mature": 0.85}

AR_PHI   = 0.85
NOISE_SD = 0.0035


def linear_trend(weeks: pd.Series, lo: float, hi: float) -> np.ndarray:
    t0 = weeks.min().value
    t1 = weeks.max().value
    span = max(1.0, t1 - t0)
    frac = (weeks.astype("int64").values - t0) / span
    return lo + (hi - lo) * frac


def ar1_noise(n: int, phi: float, sigma: float, rng) -> np.ndarray:
    e = rng.normal(0.0, sigma, size=n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


def main() -> None:
    rng   = np.random.default_rng(SEED)
    panel = pd.read_parquet(HM_INTERMEDIATE_DIR / "hm_weekly_ad.parquet")
    log.info("Input shape %s", panel.shape)

    panel = panel.sort_values(["product_id", "retailer_id", "week_start"]).reset_index(drop=True)

    for echelon in ("u_s", "u_m", "u_r"):
        out = np.zeros(len(panel))
        for prod, sub_idx in panel.groupby("product").indices.items():
            lo, hi = TREND[prod][echelon]
            trend  = linear_trend(panel.loc[sub_idx, "week_start"], lo, hi)
            bias   = panel.loc[sub_idx, "retailer"].map(RETAILER_BIAS).values
            noise  = np.zeros(len(sub_idx))
            sub_panel = panel.loc[sub_idx]
            for r_id, idx_in_sub in sub_panel.groupby("retailer_id").indices.items():
                noise[idx_in_sub] = ar1_noise(len(idx_in_sub), AR_PHI, NOISE_SD, rng)
            out[sub_idx] = np.clip(trend * bias + noise, 0.0, 0.5)
        panel[echelon] = out

    panel["avg_circ"] = panel[["u_s", "u_m", "u_r"]].mean(axis=1)

    out_path = HM_INTERMEDIATE_DIR / "hm_weekly_full.parquet"
    panel.to_parquet(out_path, index=False)
    log.info("Wrote %s shape=%s", out_path, panel.shape)

    for prod in panel["product"].unique():
        sub = panel[panel["product"] == prod]
        log.info("  %-22s u_s=%.3f  u_m=%.3f  u_r=%.3f  avg=%.3f",
                 prod, sub["u_s"].mean(), sub["u_m"].mean(),
                 sub["u_r"].mean(), sub["avg_circ"].mean())


if __name__ == "__main__":
    sys.exit(main())
