"""
SIDN-CRO -- Main experiment runner (Step 2h).

Trains and evaluates all 10 model configurations (B0-B5, A1, A2, A4) on all
4 benchmark datasets across 10 random seeds, then saves all results to JSON
and prints the paper's main results tables.

Model configurations
--------------------
    B0  AnalyticalDemand (closed-form, no learning)
    B1  MLPDemand        (2x64 ReLU, MSE)
    B2  XGBoostDemand    (400 trees, MSE)
    B3  LSTMDemand       (window=8, MSE)
    B4  BiLSTMAttnDemand (window=8, MSE, bidirectional + attention)
    B5  DRODemand        (A1 + Gaussian DRO, kappa calibrated for 90% coverage)
    A1  SIDNDemand       (structural MLP, MSE / Huber log-space)
    A2  SIDNDemand       (A1 + SPO+ DFL fine-tuning)             <- main contribution
    A4  CRODemand        (A1 + split conformal CRO)              <- robustness variant

Datasets
--------
    DataCo     2015-01 -> 2017-09  (5 products x 3 retailers)
    Olist      2016-10 -> 2018-08  (5 categories x 3 regions)
    H&M        2018-09 -> 2020-09  (5 article groups x 3 age cohorts)
    Synth-2026 2024-01 -> 2025-12  (5 products x 3 retailers, calibrated synthetic)

Runtime estimate (single-core CPU)
-----------------------------------
    B0-B2  : ~5 min total per (dataset, seed)
    B3-B4  : ~10 min per (dataset, seed)
    A1     : ~5 min per (dataset, seed)
    A2     : ~30 min per (dataset, seed)   [SQP solves dominate]
    A4     : ~10 min per (dataset, seed)   [A1 + calibration + eval]
    Total per seed: ~1 h
    All 10 seeds x 4 datasets: ~40 h  (parallelise over seeds with --seed)

Typical usage
-------------
    # Run single seed for all models on all datasets (testing / CI)
    python experiments/run_all.py --seeds 42 --skip_dfl

    # Run full paper experiment (seed 42, all models)
    python experiments/run_all.py --seeds 42

    # Run a specific seed+dataset combination
    python experiments/run_all.py --seeds 42 --datasets dataco --models B1 A1

    # Aggregate pre-saved results from all seeds and print tables
    python experiments/run_all.py --aggregate_only --results_dir results/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

# Force UTF-8 output on Windows cp1252 consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

# -- Project root on path -----------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.paths import (
    PREPROCESSED_DIR, OLIST_PREPROCESSED_DIR,
    HM_PREPROCESSED_DIR, SYNTH26_PREPROCESSED_DIR,
    RESULTS_DIR, TABLES_DIR,
)
from src.train.dataset import load_dataset, DatasetBundle
from src.train.train_baselines import train_baseline, BASELINE_CONFIGS
from src.train.train_sidn_mse import train_sidn_mse
from src.train.train_sidn_dfl import train_sidn_dfl
from src.models.conformal import build_cro_model
from src.models.dro import build_dro_model
from src.models.base import DemandModel
from src.optim.sqp_layer import params_from_panel_mean
from src.eval.metrics import (
    EvalConfig, evaluate_model_on_dataset, ModelResult,
    aggregate_seeds, print_results_table, build_pi_star_cache,
)
from src.eval.feasibility import feasibility_report, print_feasibility_table


# --------------------------------------------------------------------------- #
# Dataset registry
# --------------------------------------------------------------------------- #
DATASET_REGISTRY: Dict[str, tuple] = {
    "dataco": (PREPROCESSED_DIR,         "DataCo"),
    "olist":  (OLIST_PREPROCESSED_DIR,   "Olist"),
    "hm":     (HM_PREPROCESSED_DIR,      "H&M"),
    "synth":  (SYNTH26_PREPROCESSED_DIR, "Synth-2026"),
}

MODEL_NAMES = ["B0", "B1", "B2", "B3", "B4", "B5", "A1", "A2", "A4"]


# --------------------------------------------------------------------------- #
# Single (dataset, model, seed) run
# --------------------------------------------------------------------------- #
def run_one(
    dataset_key: str,
    model_name:  str,
    seed:        int,
    results_dir: Path,
    skip_dfl:    bool = False,
    verbose:     bool = False,
    cfg:         EvalConfig = None,
) -> Optional[Dict[str, Any]]:
    """Train one model on one dataset with one seed and evaluate.

    Returns a JSON-serialisable dict of all metrics, or None if the run
    was skipped because a cached result exists.
    """
    if cfg is None:
        cfg = EvalConfig()

    out_path = (results_dir / "raw" /
                f"{dataset_key}_{model_name}_seed{seed:04d}.json")
    if out_path.exists():
        print(f"  [SKIP] {dataset_key}/{model_name}/seed={seed} "
              f"(cached at {out_path.name})")
        with open(out_path) as f:
            return json.load(f)

    print(f"\n{'#'*70}")
    print(f"  RUN  dataset={dataset_key}  model={model_name}  seed={seed}")
    print(f"{'#'*70}")

    # -- Load dataset ---------------------------------------------------------
    prep_dir, ds_name = DATASET_REGISTRY[dataset_key]
    bundle = load_dataset(prep_dir, name=ds_name)

    # -- Build SCM parameters from training panel --------------------------
    sqp_params = params_from_panel_mean(
        bundle.train.panel_df,
        bundle.n_products,
        bundle.n_retailers,
        rng_seed=seed,
    )

    # -- Train the model ---------------------------------------------------
    t_train = time.time()
    model: Optional[DemandModel] = None
    train_metrics: Dict = {}

    if model_name in BASELINE_CONFIGS:
        model, train_metrics = train_baseline(
            model_name, bundle, seed=seed, verbose=verbose)

    elif model_name == "A1":
        model, train_metrics = train_sidn_mse(
            bundle, seed=seed, verbose=verbose)

    elif model_name == "A2":
        if skip_dfl:
            print("  [INFO] --skip_dfl: A2 replaced by A1 for this run.")
            model, train_metrics = train_sidn_mse(
                bundle, seed=seed, verbose=verbose)
            model.name = "A2_SIDNDemand_noSPO"
        else:
            a1_model, _ = train_sidn_mse(
                bundle, seed=seed, verbose=verbose)
            model, train_metrics = train_sidn_dfl(
                bundle, sqp_params,
                pretrained_sidn=a1_model,
                seed=seed, verbose=verbose)

    elif model_name == "A4":
        # A1 + conformal calibration on val set
        a1_model, train_metrics = train_sidn_mse(
            bundle, seed=seed, verbose=verbose)
        model = build_cro_model(
            a1_model,
            bundle.val.X_cont_u, bundle.val.X_cat, bundle.val.y_u,
            alpha=cfg.alpha, rho=cfg.rho,
        )
        model.name = "A4_CRODemand"

    elif model_name == "B5":
        # A1 base model + Gaussian DRO calibration on val set (comparator for A4)
        a1_model, train_metrics = train_sidn_mse(
            bundle, seed=seed, verbose=verbose)
        model = build_dro_model(
            a1_model,
            bundle.val.X_cont_u, bundle.val.X_cat, bundle.val.y_u,
            target_coverage=1.0 - cfg.alpha,   # same 90% target as A4
        )
        model.name = "B5_DRODemand"

    else:
        raise ValueError(f"Unknown model name: {model_name!r}")

    train_time = time.time() - t_train
    print(f"  Training complete in {train_time:.1f}s")

    # -- Build true-demand proxy (B0 fitted on train+val) -----------------
    from src.models.baselines import AnalyticalDemand
    true_model = AnalyticalDemand(bundle.schema, bundle.x_scaler, bundle.y_scaler)
    # Fit B0 on train+val combined (larger calibration set for pi*)
    X_tv_cont  = np.concatenate([bundle.train.X_cont_u, bundle.val.X_cont_u])
    X_tv_cat   = np.concatenate([bundle.train.X_cat,    bundle.val.X_cat])
    y_tv       = np.concatenate([bundle.train.y_u,      bundle.val.y_u])
    true_model.fit(X_tv_cont, X_tv_cat, y_tv)

    # -- Build pi* cache on test set ----------------------------------------
    from src.models.base import decision_columns as _dc
    dec_idx = _dc(bundle.schema)
    print("  Computing pi* cache for test weeks ...")
    pi_star_cache = build_pi_star_cache(
        true_model, sqp_params, bundle.test,
        bundle.n_products, bundle.n_retailers,
        dec_idx, cfg, verbose=False,
    )

    # -- Evaluate on test set ----------------------------------------------
    result = evaluate_model_on_dataset(
        model, bundle, sqp_params, true_model,
        pi_star_cache=pi_star_cache,
        seed=seed, cfg=cfg, verbose=verbose,
    )

    # -- Serialise and save ------------------------------------------------
    record = {
        "dataset":      dataset_key,
        "model":        model_name,
        "seed":         seed,
        "train_time_s": train_time,
        "train_metrics": _serialise(train_metrics),
        "summary":       result.summary(),
        "week_results": [
            {
                "week":            w.week,
                "profit_hat":      w.profit_hat,
                "profit_realised": w.profit_realised,
                "profit_star":     w.profit_star,
                "regret_abs":      w.regret_abs,
                "regret_norm":     w.regret_norm,
                "feasible":        w.feasible,
                "n_sqp_iter":      w.n_sqp_iter,
            }
            for w in result.week_results
        ],
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"  Saved: {out_path}")

    return record


# --------------------------------------------------------------------------- #
# Aggregation and table generation
# --------------------------------------------------------------------------- #
def aggregate_results(results_dir: Path, datasets: List[str],
                      models: List[str]) -> Dict:
    """Load all per-(dataset, model, seed) JSON files and aggregate by seed."""
    raw_dir = results_dir / "raw"
    all_records: Dict[str, Dict[str, List]] = {}
    # {dataset_key: {model_name: [record1, record2, ...]}}

    for ds in datasets:
        all_records[ds] = {m: [] for m in models}
        for f in sorted(raw_dir.glob(f"{ds}_*.json")):
            with open(f) as fp:
                rec = json.load(fp)
            mn = rec.get("model", "")
            if mn in models:
                all_records[ds][mn].append(rec)

    return all_records


def print_all_tables(all_records: Dict, datasets: List[str],
                     models: List[str]) -> None:
    """Print Table 2 (main results) for each dataset."""
    from src.eval.metrics import MODEL_ORDER, METRIC_COLS, print_results_table

    for ds in datasets:
        agg_by_model: Dict = {}
        for mn in models:
            recs = all_records.get(ds, {}).get(mn, [])
            if not recs:
                continue
            # Aggregate across seeds
            summaries = [r["summary"] for r in recs if "summary" in r]
            if not summaries:
                continue
            keys = summaries[0].keys()
            agg_by_model[mn] = {}
            for k in keys:
                vals = np.array([s[k] for s in summaries if k in s],
                                dtype=float)
                agg_by_model[mn][k] = {
                    "mean": float(vals.mean()),
                    "std":  float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                }

        print_results_table(agg_by_model, dataset_name=ds)

    # Summary CSV for LaTeX import
    return agg_by_model


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _serialise(obj):
    """Recursively make an object JSON-serialisable."""
    if isinstance(obj, dict):
        return {k: _serialise(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialise(v) for v in obj]
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="SIDN-CRO full experiment runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--datasets", nargs="+",
        choices=list(DATASET_REGISTRY),
        default=list(DATASET_REGISTRY),
        help="Datasets to run (default: all 4)",
    )
    parser.add_argument(
        "--models", nargs="+",
        choices=MODEL_NAMES,
        default=MODEL_NAMES,
        help="Models to train (default: all 9)",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int,
        default=list(range(42, 52)),    # seeds 42-51 (10 seeds)
        help="Random seeds (default: 42-51)",
    )
    parser.add_argument(
        "--results_dir", type=Path,
        default=RESULTS_DIR,
        help="Root directory for result files",
    )
    parser.add_argument(
        "--skip_dfl", action="store_true",
        help="Skip SPO+ DFL training for A2 (replaces with A1 predictions)",
    )
    parser.add_argument(
        "--aggregate_only", action="store_true",
        help="Skip training; only aggregate and print tables from saved results",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print per-epoch training progress",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.10,
        help="Conformal miscoverage level alpha for A4 (default 0.10 -> 90%% CI)",
    )
    parser.add_argument(
        "--rho", type=float, default=0.50,
        help="CRO robustness level rho for A4 (default 0.50)",
    )
    parser.add_argument(
        "--sqp_method", default="SLSQP", choices=["SLSQP", "trust-constr"],
        help="SQP solver method (default: SLSQP, much faster)",
    )
    args = parser.parse_args()

    cfg = EvalConfig(alpha=args.alpha, rho=args.rho, sqp_method=args.sqp_method)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # -- Aggregate-only mode -----------------------------------------------
    if args.aggregate_only:
        print("\n=== Aggregating saved results ===")
        all_records = aggregate_results(results_dir, args.datasets, args.models)
        print_all_tables(all_records, args.datasets, args.models)
        return

    # -- Training + evaluation loop ----------------------------------------
    n_total  = len(args.datasets) * len(args.models) * len(args.seeds)
    n_done   = 0
    n_failed = 0
    t_start  = time.time()

    print(f"\n{'='*70}")
    print(f"  SIDN-CRO experiment suite")
    print(f"  Datasets : {args.datasets}")
    print(f"  Models   : {args.models}")
    print(f"  Seeds    : {args.seeds}")
    print(f"  Total    : {n_total} runs")
    print(f"  skip_dfl : {args.skip_dfl}")
    print(f"{'='*70}\n")

    for seed in args.seeds:
        for dataset_key in args.datasets:
            for model_name in args.models:
                n_done += 1
                elapsed = time.time() - t_start
                eta     = (elapsed / n_done) * (n_total - n_done)
                print(f"\n[{n_done}/{n_total}]  "
                      f"elapsed={elapsed/60:.1f}min  "
                      f"ETA={eta/60:.1f}min")
                try:
                    run_one(
                        dataset_key, model_name, seed,
                        results_dir=results_dir,
                        skip_dfl=args.skip_dfl,
                        verbose=args.verbose,
                        cfg=cfg,
                    )
                except Exception as exc:
                    n_failed += 1
                    print(f"  [ERROR] {dataset_key}/{model_name}/seed={seed}: "
                          f"{exc}")
                    traceback.print_exc()

    # -- Final tables ------------------------------------------------------
    total_time = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"  Completed: {n_done - n_failed}/{n_total} runs  "
          f"({n_failed} failures)  total={total_time/60:.1f}min")
    print(f"{'='*70}\n")

    all_records = aggregate_results(results_dir, args.datasets, args.models)
    print_all_tables(all_records, args.datasets, args.models)

    # -- Save aggregated summary to JSON -----------------------------------
    summary_path = results_dir / "aggregated_summary.json"
    with open(summary_path, "w") as f:
        json.dump(_serialise(all_records), f, indent=2)
    print(f"\nAggregated summary saved to {summary_path}")


if __name__ == "__main__":
    main()
