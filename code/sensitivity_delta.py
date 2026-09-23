"""
SIDN-CRO -- Advertising-budget scale sensitivity analysis.

Tests robustness of A4 (CRODemand) prescriptive performance to changes in
the overall advertising budget relative to the main experiment's setting.

Background
----------
The main experiment builds SCMParameters via params_from_panel_mean(), which
sets the per-(i,j) ad-spend cap to b_hi=$5,000 and the per-product cooperative
ad budget (AdBudget) to 4x the maximum historical ad spend observed in the
training panel.  These are the exact values used in every run reported in the
main benchmark table.

Sweep design
------------
We define a single scalar multiplier `budget_scale` in

    {0.50, 0.75, 1.00, 1.25, 1.50}

and evaluate A4 with

    b_hi_new    = budget_scale * b_hi_base        (element-wise)
    AdBudget_new = budget_scale * AdBudget_base

where *_base are the values returned by params_from_panel_mean().
budget_scale=1.00 is the main-experiment setting (baseline); all other
multipliers correspond to +-25%/+-50% variations around it.

The pi* (oracle profit) cache is computed once at budget_scale=1.00 and
held fixed throughout the sweep.  NR% = (pi*(1.0) - pi_realised(scale)) /
pi*(1.0) therefore measures realised profit relative to the standard-budget
oracle, capturing how much headroom A4 loses or gains when the budget
assumption is changed.

Primary metric: mean NR% across 52 test weeks.  A maximum absolute change
of <1.5 pp across the full +-50% budget range is classified as LOW sensitivity.

Datasets
--------
Primary: DataCo (stable demand, most representative of the main experiment).
Cross-validation: Olist, Synth-2026.
H&M is omitted (diagnostic role in the main benchmark; NR~0% trivially).

Usage
-----
    # Single seed:
    python experiments/sensitivity_delta.py --seed 42

    # Multi-dataset:
    python experiments/sensitivity_delta.py --seed 42 --datasets dataco olist synth

    # Aggregate over seeds:
    python experiments/sensitivity_delta.py --aggregate_seeds 42 43 44 45 46

    # LaTeX table:
    python experiments/sensitivity_delta.py --seed 42 --latex

    # Save results:
    python experiments/sensitivity_delta.py --seed 42 --output results/budget_sensitivity.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Force UTF-8 output on Windows cp1252 consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.paths import (
    PREPROCESSED_DIR, OLIST_PREPROCESSED_DIR,
    HM_PREPROCESSED_DIR, SYNTH26_PREPROCESSED_DIR,
    RESULTS_DIR,
)
from src.train.dataset import load_dataset, DatasetBundle
from src.train.train_sidn_mse import train_sidn_mse
from src.models.conformal import build_cro_model
from src.models.baselines import AnalyticalDemand
from src.optim.sqp_layer import SCMParameters, params_from_panel_mean
from src.eval.metrics import (
    EvalConfig, evaluate_model_on_dataset, build_pi_star_cache,
)
from src.models.base import decision_columns


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

# The five budget-scale multipliers to sweep (1.00 = main experiment baseline)
BUDGET_SCALES: List[float] = [0.50, 0.75, 1.00, 1.25, 1.50]
BUDGET_SCALE_BASELINE: float = 1.00

# Max |delta_NR_mean| across the full sweep classified as "low sensitivity" (pp)
LOW_SENSITIVITY_THRESHOLD: float = 1.5

DATASET_REGISTRY: Dict[str, tuple] = {
    "dataco": (PREPROCESSED_DIR,         "DataCo"),
    "olist":  (OLIST_PREPROCESSED_DIR,   "Olist"),
    "hm":     (HM_PREPROCESSED_DIR,      "H&M"),
    "synth":  (SYNTH26_PREPROCESSED_DIR, "Synth-2026"),
}


# --------------------------------------------------------------------------- #
# SCMParameters builder: scale budgets around params_from_panel_mean baseline
# --------------------------------------------------------------------------- #
def params_with_budget_scale(
    panel_df,
    n_products:   int,
    n_retailers:  int,
    budget_scale: float = 1.0,
    rng_seed:     int   = 0,
) -> SCMParameters:
    """Return SCMParameters identical to params_from_panel_mean() except that
    b_hi and AdBudget are multiplied by *budget_scale*.

    budget_scale=1.0 reproduces the exact params used in the main benchmark.
    budget_scale=0.5 halves both the per-cell ad cap and the cooperative budget.
    budget_scale=1.5 increases them by 50 %.

    All other fields (prices, costs, capacity, etc.) are left unchanged.
    """
    base = params_from_panel_mean(panel_df, n_products, n_retailers,
                                  rng_seed=rng_seed)
    scaled_b_hi     = np.clip(base.b_hi     * budget_scale, 1.0, None)
    scaled_AdBudget = np.clip(base.AdBudget * budget_scale, 1.0, None)
    return replace(base, b_hi=scaled_b_hi, AdBudget=scaled_AdBudget)


# --------------------------------------------------------------------------- #
# Single-configuration evaluation
# --------------------------------------------------------------------------- #
def evaluate_budget_config(
    bundle:        DatasetBundle,
    a4_model,
    true_model,
    pi_star_cache: Optional[Dict],
    budget_scale:  float,
    seed:          int       = 42,
    cfg:           EvalConfig = None,
) -> float:
    """Evaluate A4 under *budget_scale*; return mean NR%."""
    if cfg is None:
        cfg = EvalConfig()

    sqp_params = params_with_budget_scale(
        bundle.train.panel_df,
        bundle.n_products, bundle.n_retailers,
        budget_scale=budget_scale,
        rng_seed=seed,
    )

    # pi_star_cache is fixed at budget_scale=1.0 throughout the sweep so that
    # NR% is always expressed relative to the main-experiment oracle profit.
    result = evaluate_model_on_dataset(
        a4_model, bundle, sqp_params, true_model,
        pi_star_cache=pi_star_cache,
        seed=seed, cfg=cfg, verbose=False,
    )
    return float(result.summary().get("NR_mean", float("nan")))


# --------------------------------------------------------------------------- #
# Full sweep: one (dataset, seed)
# --------------------------------------------------------------------------- #
def run_budget_sweep(
    dataset_key: str,
    seed:        int       = 42,
    cfg:         EvalConfig = None,
    verbose:     bool      = False,
) -> Dict:
    """Sweep budget_scale for one (dataset, seed).

    Returns
    -------
    dict with keys:
        "dataset"     : dataset_key
        "seed"        : seed
        "baseline_NR" : NR% at budget_scale=1.0
        "sweep"       : [ {scale, NR_pct, delta_NR_pp} ]
    """
    if cfg is None:
        cfg = EvalConfig()

    prep_dir, ds_name = DATASET_REGISTRY[dataset_key]
    bundle = load_dataset(prep_dir, name=ds_name)

    print(f"\n{'='*72}")
    print(f"  budget_scale sensitivity: dataset={dataset_key}  seed={seed}")
    print(f"{'='*72}")

    # -- Train A4 once --------------------------------------------------------
    print("  [1/4] Training A4 (SIDN-MSE + CRO calibration)...")
    t0 = time.time()
    a1_model, _ = train_sidn_mse(bundle, seed=seed, verbose=False)
    a4_model    = build_cro_model(
        a1_model,
        bundle.val.X_cont_u, bundle.val.X_cat, bundle.val.y_u,
        alpha=cfg.alpha, rho=cfg.rho,
    )
    a4_model.name = "A4_budget_sens"
    print(f"  A4 ready in {time.time()-t0:.1f}s")

    # -- True-demand proxy (B0 on train+val) ----------------------------------
    print("  [2/4] Fitting true-demand proxy (B0 on train+val)...")
    true_model = AnalyticalDemand(bundle.schema, bundle.x_scaler, bundle.y_scaler)
    X_tv_cont  = np.concatenate([bundle.train.X_cont_u, bundle.val.X_cont_u])
    X_tv_cat   = np.concatenate([bundle.train.X_cat,    bundle.val.X_cat])
    y_tv       = np.concatenate([bundle.train.y_u,      bundle.val.y_u])
    true_model.fit(X_tv_cont, X_tv_cat, y_tv)

    # -- pi* cache (computed once at budget_scale=1.0, the main experiment) ---
    print(f"  [3/4] Computing pi* cache at budget_scale={BUDGET_SCALE_BASELINE:.2f} "
          f"(main-experiment baseline)...")
    sqp_base  = params_with_budget_scale(
        bundle.train.panel_df,
        bundle.n_products, bundle.n_retailers,
        budget_scale=BUDGET_SCALE_BASELINE,
        rng_seed=seed,
    )
    dec_idx = decision_columns(bundle.schema)
    pi_star_cache = build_pi_star_cache(
        true_model, sqp_base, bundle.test,
        bundle.n_products, bundle.n_retailers,
        dec_idx, cfg, verbose=False,
    )

    # -- Sweep budget_scale ---------------------------------------------------
    print("  [4/4] Sweeping budget_scale configurations...")
    sweep: List[Dict] = []
    nr_baseline: float = float("nan")

    for scale in BUDGET_SCALES:
        nr = evaluate_budget_config(
            bundle, a4_model, true_model, pi_star_cache,
            budget_scale=scale, seed=seed, cfg=cfg,
        )
        is_baseline = abs(scale - BUDGET_SCALE_BASELINE) < 1e-9
        if is_baseline:
            nr_baseline = nr

        # delta_NR is filled below once nr_baseline is known (always at 1.0)
        sweep.append({"scale": float(scale), "NR_pct": nr})

    # Fill delta_NR now (nr_baseline guaranteed computed since 1.0 is in list)
    for pt in sweep:
        pt["delta_NR_pp"] = pt["NR_pct"] - nr_baseline
        marker = " <-- baseline" if abs(pt["scale"] - BUDGET_SCALE_BASELINE) < 1e-9 else ""
        print(f"    x{pt['scale']:.2f}  NR%={pt['NR_pct']:.2f}  "
              f"dNR={pt['delta_NR_pp']:+.2f}pp{marker}")

    print(f"  Baseline NR% = {nr_baseline:.2f}%  (budget_scale={BUDGET_SCALE_BASELINE:.2f})")

    return {
        "dataset":     dataset_key,
        "seed":        seed,
        "baseline_NR": nr_baseline,
        "sweep":       sweep,
    }


# --------------------------------------------------------------------------- #
# Multi-seed aggregation
# --------------------------------------------------------------------------- #
def aggregate_budget_sweeps(records: List[Dict]) -> Dict:
    """Aggregate sweep results over multiple seeds.

    Returns
    -------
    dict {
        "baseline_NR_mean": float,
        "baseline_NR_std":  float,
        "n_seeds":          int,
        "aggregate": [ {scale, NR_mean, NR_std, dNR_mean} ]
    }
    """
    baseline_nrs  = [r["baseline_NR"] for r in records]
    baseline_mean = float(np.mean(baseline_nrs))
    baseline_std  = (float(np.std(baseline_nrs, ddof=1))
                     if len(baseline_nrs) > 1 else 0.0)

    # Collect NR% per scale across seeds
    from collections import defaultdict
    nr_by_scale: Dict[float, List[float]] = defaultdict(list)
    for rec in records:
        for pt in rec["sweep"]:
            nr_by_scale[pt["scale"]].append(pt["NR_pct"])

    agg: List[Dict] = []
    for scale in sorted(nr_by_scale):
        vals = np.array(nr_by_scale[scale])
        agg.append({
            "scale":    float(scale),
            "NR_mean":  float(vals.mean()),
            "NR_std":   float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            "dNR_mean": float(vals.mean() - baseline_mean),
        })

    return {
        "baseline_NR_mean": baseline_mean,
        "baseline_NR_std":  baseline_std,
        "n_seeds":          len(records),
        "aggregate":        agg,
    }


# --------------------------------------------------------------------------- #
# Print helpers
# --------------------------------------------------------------------------- #
def print_sweep_table(rec: Dict) -> None:
    """Print single-seed budget_scale sweep table."""
    ds      = rec["dataset"]
    seed    = rec["seed"]
    nr_base = rec["baseline_NR"]
    sweep   = rec["sweep"]

    max_abs = max(abs(p["delta_NR_pp"]) for p in sweep)
    sensitivity = ("LOW"      if max_abs < LOW_SENSITIVITY_THRESHOLD else
                   "MODERATE" if max_abs < 3.0 else "HIGH")

    print(f"\n{'='*66}")
    print(f"  budget_scale sensitivity  |  dataset={ds}  seed={seed}")
    print(f"  Baseline NR% = {nr_base:.2f}%  "
          f"(threshold: +-{LOW_SENSITIVITY_THRESHOLD}pp)")
    print(f"  Max |dNR| = {max_abs:.2f}pp  [{sensitivity}]")
    print(f"{'='*66}")
    print(f"  {'Scale':>6}  {'NR%':>8}  {'dNR (pp)':>10}")
    print(f"  {'-'*28}")
    for pt in sweep:
        mark = " <--" if abs(pt["scale"] - BUDGET_SCALE_BASELINE) < 1e-9 else "    "
        print(f"  x{pt['scale']:.2f}  {pt['NR_pct']:>8.2f}  "
              f"{pt['delta_NR_pp']:>+10.2f}pp{mark}")


def print_aggregate_table(agg_rec: Dict, dataset_key: str) -> None:
    """Print aggregated (multi-seed) budget_scale sweep table."""
    nr_base  = agg_rec["baseline_NR_mean"]
    nr_std   = agg_rec["baseline_NR_std"]
    n_seeds  = agg_rec["n_seeds"]
    agg      = agg_rec["aggregate"]

    max_abs = max(abs(p["dNR_mean"]) for p in agg)
    sensitivity = ("LOW"      if max_abs < LOW_SENSITIVITY_THRESHOLD else
                   "MODERATE" if max_abs < 3.0 else "HIGH")

    print(f"\n{'='*72}")
    print(f"  budget_scale sensitivity (N={n_seeds} seeds)  |  dataset={dataset_key}")
    print(f"  Baseline NR% = {nr_base:.2f} +/- {nr_std:.2f}%  "
          f"(threshold: +-{LOW_SENSITIVITY_THRESHOLD}pp)")
    print(f"  Max |dNR_mean| = {max_abs:.2f}pp  [{sensitivity}]")
    print(f"{'='*72}")
    print(f"  {'Scale':>6}  {'NR_mean':>9}  {'NR_std':>8}  {'dNR_mean (pp)':>14}")
    print(f"  {'-'*44}")
    for pt in agg:
        mark = " <--" if abs(pt["scale"] - BUDGET_SCALE_BASELINE) < 1e-9 else "    "
        print(f"  x{pt['scale']:.2f}  {pt['NR_mean']:>9.2f}  {pt['NR_std']:>8.2f}  "
              f"{pt['dNR_mean']:>+14.2f}pp{mark}")


def print_latex_table(agg_rec: Dict, dataset_key: str = "") -> None:
    """Print LaTeX tabular for manuscript sensitivity appendix.

    Produces a single-column sweep table:
        Scale | NR_mean (%) | delta_NR_mean (pp)
    """
    nr_base = agg_rec["baseline_NR_mean"]
    n_seeds = agg_rec["n_seeds"]
    agg     = agg_rec["aggregate"]

    cap = f"Budget-scale sensitivity for {dataset_key} (N={n_seeds} seeds)." if dataset_key else ""

    print()
    print(r"\begin{tabular}{rrr}")
    print(r"\toprule")
    print(r"Budget scale & $\overline{\mathrm{NR}}$ (\%) "
          r"& $\Delta\overline{\mathrm{NR}}$ (pp) \\")
    print(r"\midrule")
    for pt in agg:
        scale   = pt["scale"]
        sc_str  = f"${scale:.2f}\\times$"
        nr_str  = f"{pt['NR_mean']:.2f}"
        dnr_str = f"{pt['dNR_mean']:+.2f}"
        bold    = abs(scale - BUDGET_SCALE_BASELINE) < 1e-9
        if bold:
            sc_str  = r"\textbf{" + sc_str  + "}"
            nr_str  = r"\textbf{" + nr_str  + "}"
            dnr_str = r"\textbf{" + dnr_str + "}"
        print(f"{sc_str} & {nr_str} & {dnr_str} \\\\")
    print(r"\midrule")
    print(r"\multicolumn{3}{l}{\textit{Baseline} $\overline{\mathrm{NR}} = "
          f"{nr_base:.2f}$\\% (N={n_seeds} seeds)" + r"} \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    if cap:
        print(f"% Caption: {cap}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="Advertising-budget scale sensitivity analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--datasets", nargs="+",
        choices=list(DATASET_REGISTRY),
        default=["dataco"],
        help="Datasets to analyse (default: dataco)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Single random seed (default: 42)",
    )
    parser.add_argument(
        "--aggregate_seeds", nargs="+", type=int, default=None,
        help="Run over these seeds and aggregate (overrides --seed)",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.10,
        help="Conformal alpha for A4 (default 0.10 -> 90%% coverage)",
    )
    parser.add_argument(
        "--rho", type=float, default=0.50,
        help="CRO robustness rho for A4 (default 0.50)",
    )
    parser.add_argument(
        "--latex", action="store_true",
        help="Also print LaTeX table for manuscript appendix",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Save full results to this JSON path",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print per-week SQP details",
    )
    args = parser.parse_args()

    cfg   = EvalConfig(alpha=args.alpha, rho=args.rho)
    seeds = args.aggregate_seeds if args.aggregate_seeds else [args.seed]

    all_outputs: Dict[str, object] = {}
    for ds in args.datasets:
        records = []
        for seed in seeds:
            rec = run_budget_sweep(ds, seed=seed, cfg=cfg, verbose=args.verbose)
            print_sweep_table(rec)
            records.append(rec)

        if len(records) > 1:
            agg = aggregate_budget_sweeps(records)
            print_aggregate_table(agg, ds)
            if args.latex:
                print(f"\n% LaTeX sensitivity table -- {ds}")
                print_latex_table(agg, dataset_key=ds)
            all_outputs[ds] = agg
        else:
            rec = records[0]
            all_outputs[ds] = rec
            if args.latex:
                fake_agg = {
                    "baseline_NR_mean": rec["baseline_NR"],
                    "baseline_NR_std":  0.0,
                    "n_seeds":          1,
                    "aggregate": [
                        {
                            "scale":    pt["scale"],
                            "NR_mean":  pt["NR_pct"],
                            "NR_std":   0.0,
                            "dNR_mean": pt["delta_NR_pp"],
                        }
                        for pt in rec["sweep"]
                    ],
                }
                print(f"\n% LaTeX sensitivity table -- {ds}")
                print_latex_table(fake_agg, dataset_key=ds)

    if args.output:
        def _ser(v):
            if isinstance(v, np.floating):  return float(v)
            if isinstance(v, np.integer):   return int(v)
            if isinstance(v, np.bool_):     return bool(v)
            if isinstance(v, np.ndarray):   return v.tolist()
            return v

        import json as _json
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            _json.dump(all_outputs, f, indent=2, default=_ser)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
