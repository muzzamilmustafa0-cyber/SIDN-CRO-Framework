"""
SIDN-CRO data pipeline -- Cross-replication (Stage 3 / Olist).

Generates calibrated circular-input fractions u_s, u_m, u_r for the Olist
panel using the same modelling assumptions as the DataCo branch but with
Brazilian-sectoral constants:

* Linear trends per (product, echelon) calibrated against Brazilian circular-
  economy reports (CNI 2019; CETESB 2020; SUDENE Circular Economy Atlas):
        bed_bath_table          (textile-like, slow growth)
        health_beauty           (recycled packaging, moderate)
        sports_leisure          (sports merchandise, slow)
        furniture_decor         (recycled wood / metals, slow)
        computers_accessories   (e-waste recovery, moderate)

* Within each product the ranking u_s < u_m < u_r is preserved.

* Retailer-region bias:
        SP        1.20  (more progressive sustainability policies)
        Southeast 1.00
        Other     0.80  (least developed circular infrastructure)

* AR(1) noise (phi=0.85, sigma=0.0035) per (product, retailer) series.

Random seed 43.
"""

from __future__ import annotations
import logging
import sys
import numpy as np
import pandas as pd

from src.utils.paths import OLIST_INTERMEDIATE_DIR

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("synth_circ_olist")

SEED = 43

# (start_2016Q3, end_2018Q3) trends
TREND = {
    "bed_bath_table":        {"u_s": (0.005, 0.018), "u_m": (0.010, 0.030), "u_r": (0.015, 0.045)},
    "health_beauty":         {"u_s": (0.007, 0.022), "u_m": (0.014, 0.035), "u_r": (0.020, 0.050)},
    "sports_leisure":        {"u_s": (0.003, 0.012), "u_m": (0.005, 0.020), "u_r": (0.008, 0.030)},
    "furniture_decor":       {"u_s": (0.004, 0.015), "u_m": (0.008, 0.025), "u_r": (0.012, 0.038)},
    "computers_accessories": {"u_s": (0.006, 0.020), "u_m": (0.012, 0.032), "u_r": (0.018, 0.045)},
}

RETAILER_BIAS = {"SP": 1.20, "Southeast": 1.00, "Other": 0.80}

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
    panel = pd.read_parquet(OLIST_INTERMEDIATE_DIR / "olist_weekly_ad.parquet")
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

    out_path = OLIST_INTERMEDIATE_DIR / "olist_weekly_full.parquet"
    panel.to_parquet(out_path, index=False)
    log.info("Wrote %s shape=%s", out_path, panel.shape)

    for prod in panel["product"].unique():
        sub = panel[panel["product"] == prod]
        log.info("  %-25s u_s=%.3f  u_m=%.3f  u_r=%.3f  avg=%.3f",
                 prod, sub["u_s"].mean(), sub["u_m"].mean(),
                 sub["u_r"].mean(), sub["avg_circ"].mean())


if __name__ == "__main__":
    sys.exit(main())
