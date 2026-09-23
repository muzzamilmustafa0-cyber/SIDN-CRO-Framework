"""
SIDN-CRO -- Fast experiment runner (single seed, shared pi* cache).

Builds the pi* cache ONCE per dataset, then evaluates all models efficiently.
Each (dataset, model) result is cached to JSON. Skips existing results.

Usage:
    python experiments/fast_run.py --seed 42 --sqp_method SLSQP
    python experiments/fast_run.py --datasets dataco --models B0 B1 B2 A1 A4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import warnings
warnings.filterwarnings("ignore")

from src.utils.paths import (
    PREPROCESSED_DIR, OLIST_PREPROCESSED_DIR,
    HM_PREPROCESSED_DIR, SYNTH26_PREPROCESSED_DIR,
    RESULTS_DIR,
)
from src.train.dataset import load_dataset, DatasetBundle
from src.train.train_baselines import train_baseline, BASELINE_CONFIGS
from src.train.train_sidn_mse import train_sidn_mse
from src.models.conformal import build_cro_model
from src.models.baselines import AnalyticalDemand
from src.models.base import decision_columns
from src.optim.sqp_layer import params_from_panel_mean
from src.eval.metrics import (
    EvalConfig, evaluate_model_on_dataset, build_pi_star_cache,
)

DATASET_REGISTRY = {
    "dataco": (PREPROCESSED_DIR,         "DataCo"),
    "olist":  (OLIST_PREPROCESSED_DIR,   "Olist"),
    "hm":     (HM_PREPROCESSED_DIR,      "H&M"),
    "synth":  (SYNTH26_PREPROCESSED_DIR, "Synth-2026"),
}
MODEL_ORDER = ["B0", "B1", "B2", "B3", "B4", "A1", "A2", "A4"]


def _serialise(obj):
    if isinstance(obj, dict):   return {k: _serialise(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [_serialise(v) for v in obj]
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    return obj


def run_dataset(
    dataset_key: str,
    models: List[str],
    seed: int,
    results_dir: Path,
    cfg: EvalConfig,
    verbose: bool = False,
    skip_dfl: bool = True,
):
    """Run all models on one dataset, sharing the pi* cache."""
    prep_dir, ds_name = DATASET_REGISTRY[dataset_key]
    print(f"\n{'#'*70}")
    print(f"  DATASET: {ds_name}  seed={seed}  models={models}")
    print(f"{'#'*70}")

    # -- Load dataset ──────────────────────────────────────────────────────────
    bundle = load_dataset(prep_dir, name=ds_name)
    sqp_params = params_from_panel_mean(
        bundle.train.panel_df, bundle.n_products, bundle.n_retailers,
        rng_seed=seed,
    )
    dec_idx = decision_columns(bundle.schema)

    # -- True demand proxy (B0 on train+val) ───────────────────────────────────
    print(f"\n  Building true demand proxy (B0 on train+val)...")
    true_model = AnalyticalDemand(bundle.schema, bundle.x_scaler, bundle.y_scaler)
    X_tv = np.concatenate([bundle.train.X_cont_u, bundle.val.X_cont_u])
    C_tv = np.concatenate([bundle.train.X_cat,    bundle.val.X_cat])
    y_tv = np.concatenate([bundle.train.y_u,      bundle.val.y_u])
    true_model.fit(X_tv, C_tv, y_tv)
    print(f"  True demand proxy: done")

    # -- Build pi* cache ONCE for all models ───────────────────────────────────
    print(f"\n  Building pi* cache (shared across all models)...")
    t0 = time.time()
    pi_cache = build_pi_star_cache(
        true_model, sqp_params, bundle.test,
        bundle.n_products, bundle.n_retailers,
        dec_idx, cfg, verbose=True,
    )
    print(f"  pi* cache: {len(pi_cache)} weeks in {time.time()-t0:.1f}s")

    # -- Train and evaluate each model ─────────────────────────────────────────
    a1_model = None   # cache A1 for A4

    for model_name in models:
        out_path = results_dir / "raw" / f"{dataset_key}_{model_name}_seed{seed:04d}.json"
        if out_path.exists():
            print(f"\n  [SKIP] {dataset_key}/{model_name}/seed={seed} (cached)")
            # Still load A1 if needed for A4
            if model_name == "A1" and a1_model is None:
                a1_model, _ = train_sidn_mse(bundle, seed=seed, verbose=verbose)
            continue

        print(f"\n  --- {model_name} ---")
        t_train = time.time()
        model = None

        try:
            if model_name in BASELINE_CONFIGS:
                model, train_m = train_baseline(model_name, bundle, seed=seed, verbose=verbose)

            elif model_name == "A1":
                model, train_m = train_sidn_mse(bundle, seed=seed, verbose=verbose)
                a1_model = model

            elif model_name == "A2":
                if skip_dfl:
                    model, train_m = train_sidn_mse(bundle, seed=seed, verbose=verbose)
                    model.name = "A2_SIDNDemand_noSPO"
                else:
                    from src.train.train_sidn_dfl import train_sidn_dfl
                    if a1_model is None:
                        a1_model, _ = train_sidn_mse(bundle, seed=seed, verbose=verbose)
                    model, train_m = train_sidn_dfl(
                        bundle, sqp_params, pretrained_sidn=a1_model,
                        seed=seed, verbose=verbose,
                        sqp_method=cfg.sqp_method,   # use SLSQP for speed
                        sqp_tol=1e-3,                # loose tol for DFL inner loop
                        sqp_max_iter=50,             # fast inner solves
                        dfl_epochs=10,               # sufficient for fine-tuning
                    )

            elif model_name == "A4":
                if a1_model is None:
                    a1_model, train_m = train_sidn_mse(bundle, seed=seed, verbose=verbose)
                else:
                    train_m = {}
                model = build_cro_model(
                    a1_model,
                    bundle.val.X_cont_u, bundle.val.X_cat, bundle.val.y_u,
                    alpha=cfg.alpha, rho=cfg.rho,
                )
                model.name = "A4_CRODemand"

            else:
                print(f"  [WARN] Unknown model {model_name!r}; skipping")
                continue

            train_time = time.time() - t_train

            # Evaluate
            result = evaluate_model_on_dataset(
                model, bundle, sqp_params, true_model,
                pi_star_cache=pi_cache,
                seed=seed, cfg=cfg, verbose=verbose,
            )

            # Save
            record = {
                "dataset":       dataset_key,
                "model":         model_name,
                "seed":          seed,
                "train_time_s":  train_time,
                "train_metrics": _serialise(train_m if isinstance(train_m, dict) else {}),
                "summary":       result.summary(),
                "week_results": [
                    {k: getattr(w, k) for k in
                     ["week","profit_hat","profit_realised","profit_star",
                      "regret_abs","regret_norm","feasible","n_sqp_iter"]}
                    for w in result.week_results
                ],
            }
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(record, f, indent=2)
            print(f"  Saved: {out_path.name}  (train={train_time:.0f}s)")

        except Exception as exc:
            print(f"  [ERROR] {dataset_key}/{model_name}: {exc}")
            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="Fast SIDN-CRO experiment runner")
    parser.add_argument("--datasets", nargs="+",
                        choices=list(DATASET_REGISTRY),
                        default=list(DATASET_REGISTRY))
    parser.add_argument("--models", nargs="+",
                        choices=MODEL_ORDER,
                        default=["B0","B1","B2","A1","A4"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sqp_method", default="SLSQP",
                        choices=["SLSQP","trust-constr"])
    parser.add_argument("--run_dfl", action="store_true", default=False,
                        help="Run A2 with real SPO+ DFL training (slow, ~20 min/dataset)")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--results_dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    # skip_dfl=False means actually run SPO+ fine-tuning
    args.skip_dfl = not args.run_dfl

    cfg = EvalConfig(sqp_method=args.sqp_method)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "raw").mkdir(exist_ok=True)

    t_start = time.time()
    print(f"\n{'='*70}")
    print(f"  Fast SIDN-CRO Runner")
    print(f"  Datasets: {args.datasets}")
    print(f"  Models  : {args.models}")
    print(f"  Seed    : {args.seed}")
    print(f"  Method  : {args.sqp_method}")
    print(f"{'='*70}")

    for ds in args.datasets:
        try:
            run_dataset(
                ds, args.models, args.seed,
                results_dir, cfg,
                verbose=args.verbose,
                skip_dfl=args.skip_dfl,
            )
        except Exception as e:
            print(f"[ERROR] Dataset {ds}: {e}")
            traceback.print_exc()

    elapsed = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"  Done in {elapsed/60:.1f} min")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
