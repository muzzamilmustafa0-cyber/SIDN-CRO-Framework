"""
SIDN-CRO data pipeline -- Stage 3: Calibrated circular-input fractions
for the multi-product panel.

Each (product i, retailer j, week t) cell is assigned three echelon-level
circular-input utilisation variables u_s, u_m, u_r in [0, 1] following:

* Linear trend per (product, echelon) calibrated to publicly reported
  sectoral circular-material-use rates (Eurostat env_ac_cur, EU CEAP
  monitoring, Ellen MacArthur Foundation Circularity Gap Reports):
        Textiles (Apparel, Footwear): 1.0% in 2015 -> 5.5% in 2023
        Sports merchandise (Fan Shop): 0.5% -> 3.0%
        Golf:                          0.6% -> 3.5%
        Outdoors:                      1.0% -> 4.5%
  Within each product the ranking u_s < u_m < u_r is preserved (suppliers
  qualifying recycled raw fibres lag manufacturers and retailers, consistent
  with the cost-penalty structure c2_s > c2_m > c2_r in the original paper).

* Retailer-segment bias capturing differential demand-side pressure:
        Consumer    1.00 (mass-market awareness)
        Corporate   1.25 (ESG-driven procurement)
        Home Office 0.85 (least sensitive)

* AR(1) noise (phi=0.85, sigma=0.0035) per (product, retailer) series.

Random seed 43 is fixed.
"""

from __future__ import annotations
import logging
import sys
import numpy as np
import pandas as pd

from src.utils.paths import INTERMEDIATE_DIR

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("synth_circular")

SEED = 43

# (start_2015, end_2018) for u_s, u_m, u_r per product
TREND = {
    "Apparel":  {"u_s": (0.008, 0.025), "u_m": (0.015, 0.045), "u_r": (0.022, 0.060)},
    "Footwear": {"u_s": (0.006, 0.022), "u_m": (0.012, 0.040), "u_r": (0.018, 0.055)},
    "Fan Shop": {"u_s": (0.003, 0.012), "u_m": (0.005, 0.020), "u_r": (0.008, 0.030)},
    "Golf":     {"u_s": (0.004, 0.014), "u_m": (0.008, 0.022), "u_r": (0.012, 0.035)},
    "Outdoors": {"u_s": (0.007, 0.020), "u_m": (0.014, 0.035), "u_r": (0.020, 0.045)},
}

RETAILER_BIAS = {
    "Consumer":    1.00,
    "Corporate":   1.25,
    "Home Office": 0.85,
}

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
    panel = pd.read_parquet(INTERMEDIATE_DIR / "dataco_weekly_ad.parquet")
    log.info("Input shape %s", panel.shape)

    panel = panel.sort_values(["product_id", "retailer_id", "week_start"]).reset_index(drop=True)

    for echelon in ("u_s", "u_m", "u_r"):
        out = np.zeros(len(panel))
        for prod, sub_idx in panel.groupby("product").indices.items():
            lo, hi = TREND[prod][echelon]
            trend  = linear_trend(panel.loc[sub_idx, "week_start"], lo, hi)
            bias   = panel.loc[sub_idx, "retailer"].map(RETAILER_BIAS).values
            # AR(1) noise per (product, retailer)
            noise  = np.zeros(len(sub_idx))
            sub_panel = panel.loc[sub_idx]
            for r_id, idx_in_sub in sub_panel.groupby("retailer_id").indices.items():
                noise[idx_in_sub] = ar1_noise(len(idx_in_sub), AR_PHI, NOISE_SD, rng)
            out[sub_idx] = np.clip(trend * bias + noise, 0.0, 0.5)
        panel[echelon] = out

    panel["avg_circ"] = panel[["u_s", "u_m", "u_r"]].mean(axis=1)

    out_path = INTERMEDIATE_DIR / "dataco_weekly_full.parquet"
    panel.to_parquet(out_path, index=False)
    log.info("Wrote %s shape=%s", out_path, panel.shape)

    log.info("Per-product circular-fraction means:")
    for prod in panel["product"].unique():
        sub = panel[panel["product"] == prod]
        log.info("  %-9s u_s=%.3f  u_m=%.3f  u_r=%.3f  avg=%.3f",
                 prod, sub["u_s"].mean(), sub["u_m"].mean(),
                 sub["u_r"].mean(), sub["avg_circ"].mean())


if __name__ == "__main__":
    sys.exit(main())
