"""
SIDN-CRO -- Sensitivity analysis (Step 2g, part 2).

Three sensitivity experiments reported in the manuscript:

1.  Robustness-level sensitivity (Figure 6a):
        Vary rho in {0, 0.25, 0.50, 0.75, 1.0} in CRODemand and measure
        normalised regret vs. coverage trade-off.

2.  Context-window sensitivity (Figure 6b):
        Vary LSTM / Bi-LSTM window size T in {4, 8, 12, 16} and compare
        demand MAPE and normalised regret.

3.  Distribution-shift severity (Figure 7):
        Evaluate all models on test subsets with increasing covariate shift
        (ordered by quartile of |price_test ? price_train_mean|) and track
        NR as a function of shift magnitude.
"""

from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np

from src.models.base import DemandModel, decision_columns
from src.models.conformal import CRODemand
from src.optim.sqp_layer import SCMParameters
from src.train.dataset import DatasetBundle, SplitArrays, compute_demand_metrics
from src.eval.metrics import EvalConfig, ModelResult, evaluate_model_on_dataset


# --------------------------------------------------------------------------- #
# 1. Robustness-level sensitivity (rho sweep for A4 / CRODemand)
# --------------------------------------------------------------------------- #
def rho_sensitivity_table(
    base_model:      DemandModel,
    bundle:          DatasetBundle,
    sqp_params:      SCMParameters,
    true_demand_fn:  DemandModel,
    pi_star_cache:   Dict[int, float],
    rho_grid:        Optional[List[float]] = None,
    alpha:           float = 0.10,
    seed:            int   = 42,
    cfg:             EvalConfig = None,
) -> Dict[float, Dict[str, float]]:
    """Evaluate A4 (CRODemand) at multiple robustness levels rho.

    Parameters
    ----------
    base_model    : fitted A1 or A2 model to wrap with CRO
    rho_grid      : list of rho values to sweep (default: [0, 0.25, 0.5, 0.75, 1.0])
    alpha         : conformal miscoverage level
    pi_star_cache : pre-computed pi* per test week (avoids re-solving)

    Returns
    -------
    { rho: { "NR_mean": ..., "NR_std": ..., "coverage": ..., "demand_mape": ... } }
    """
    if rho_grid is None:
        rho_grid = [0.0, 0.25, 0.50, 0.75, 1.00]
    if cfg is None:
        cfg = EvalConfig()

    # Calibrate conformal intervals once
    val = bundle.val
    y_pred_cal = base_model.raw_predict(val.X_cont_u, val.X_cat)

    results: Dict[float, Dict] = {}
    for rho in rho_grid:
        cro = CRODemand(base_model, alpha=alpha, rho=rho)
        cro.calibrate(val.X_cont_u, val.X_cat, val.y_u)

        mr = evaluate_model_on_dataset(
            cro, bundle, sqp_params, true_demand_fn,
            pi_star_cache=pi_star_cache,
            seed=seed, cfg=cfg, verbose=False,
        )
        s = mr.summary()
        # One-sided coverage: Pr(D_true >= D_robust)
        D_hat = base_model.raw_predict(bundle.test.X_cont_u, bundle.test.X_cat)
        q = (cro.calibrator.q_lo if cro.calibrator.asymmetric
             else cro.calibrator.q_sym)
        D_robust = np.maximum(0.0, D_hat - rho * q)
        coverage = float((bundle.test.y_u >= D_robust).mean())

        results[rho] = {
            "NR_mean":    s.get("NR_mean", 0.0),
            "NR_std":     s.get("NR_std", 0.0),
            "coverage":   coverage * 100,
            "demand_mape": s.get("demand_mape", 0.0),
        }
        print(f"  rho={rho:.2f}  NR={results[rho]['NR_mean']:.2f}%  "
              f"coverage={results[rho]['coverage']:.1f}%")

    return results


# --------------------------------------------------------------------------- #
# 2. LSTM window-size sensitivity
# --------------------------------------------------------------------------- #
def window_sensitivity_table(
    bundle:         DatasetBundle,
    sqp_params:     SCMParameters,
    true_demand_fn: DemandModel,
    pi_star_cache:  Dict[int, float],
    window_grid:    Optional[List[int]] = None,
    model_cls_name: str = "B3",   # "B3" (LSTM) or "B4" (BiLSTM)
    seed:           int = 42,
    cfg:            EvalConfig = None,
) -> Dict[int, Dict[str, float]]:
    """Train LSTM/BiLSTM with different window sizes and compare NR.

    Returns
    -------
    { window: {"NR_mean": ..., "demand_mape": ..., "fit_time_s": ...} }
    """
    from src.train.train_baselines import BASELINE_CONFIGS, train_baseline

    if window_grid is None:
        window_grid = [4, 8, 12, 16]
    if cfg is None:
        cfg = EvalConfig()

    base_cfg = BASELINE_CONFIGS[model_cls_name]
    results: Dict[int, Dict] = {}

    for T in window_grid:
        print(f"\n  Window T={T} ...")
        # Override window size in the init_kw
        import copy
        mod_cfg = copy.deepcopy(base_cfg)
        mod_cfg["init_kw"]["window"] = T

        # Inline training (bypass train_baseline to override init_kw cleanly)
        cls   = mod_cfg["cls"]
        model = cls(bundle.schema, bundle.x_scaler, bundle.y_scaler,
                    **mod_cfg["init_kw"])
        tr    = bundle.train
        model.fit(tr.X_cont, tr.X_cat, tr.y,
                  week_idx=tr.week_idx, seed=seed,
                  **mod_cfg["fit_kw"])

        mr = evaluate_model_on_dataset(
            model, bundle, sqp_params, true_demand_fn,
            pi_star_cache=pi_star_cache,
            seed=seed, cfg=cfg, verbose=False,
        )
        s = mr.summary()
        results[T] = {
            "NR_mean":     s.get("NR_mean", 0.0),
            "NR_std":      s.get("NR_std", 0.0),
            "demand_mape": s.get("demand_mape", 0.0),
        }
        print(f"  T={T}  NR={results[T]['NR_mean']:.2f}%  "
              f"MAPE={results[T]['demand_mape']:.2f}%")

    return results


# --------------------------------------------------------------------------- #
# 3. Distribution-shift severity
# --------------------------------------------------------------------------- #
def dataset_shift_table(
    models:         Dict[str, DemandModel],
    bundle:         DatasetBundle,
    sqp_params:     SCMParameters,
    true_demand_fn: DemandModel,
    pi_star_cache:  Dict[int, float],
    n_quartiles:    int = 4,
    shift_feature:  str = "p",      # cont feature to measure shift by
    cfg:            EvalConfig = None,
) -> Dict[str, Dict[int, float]]:
    """Evaluate all models on test sub-sets grouped by covariate shift quartile.

    Covariate shift is measured as |feature_test ? feature_train_mean| for the
    specified ``shift_feature`` (default: price 'p').  The test set is split
    into quartiles of this shift magnitude.

    Returns
    -------
    { model_name: { quartile_idx: NR_mean } }
    Quartile 1 = lowest shift (? in-distribution), quartile 4 = highest shift.
    """
    if cfg is None:
        cfg = EvalConfig()

    dec = decision_columns(bundle.schema)
    feat_idx = dec.get(shift_feature, 0)

    # Compute shift magnitude per test row
    train_mean = float(bundle.train.X_cont_u[:, feat_idx].mean())
    shift_mag  = np.abs(bundle.test.X_cont_u[:, feat_idx] - train_mean)

    # Assign quartile labels
    quartile_thresholds = np.quantile(shift_mag, np.linspace(0, 1, n_quartiles + 1))
    quartile_labels     = np.digitize(shift_mag, quartile_thresholds[1:-1])

    results: Dict[str, Dict[int, float]] = {m: {} for m in models}

    for q_idx in range(n_quartiles):
        mask = quartile_labels == q_idx
        n_q  = mask.sum()
        if n_q < bundle.n_products * bundle.n_retailers:
            continue   # not enough rows for a complete week
        print(f"\n  Quartile {q_idx+1}/{n_quartiles}  "
              f"n={n_q}  "
              f"shift_range=[{quartile_thresholds[q_idx]:.2f}, "
              f"{quartile_thresholds[q_idx+1]:.2f}]")

        # Subset test split
        sub = SplitArrays(
            X_cont   = bundle.test.X_cont[mask],
            X_cat    = bundle.test.X_cat[mask],
            y        = bundle.test.y[mask],
            X_cont_u = bundle.test.X_cont_u[mask],
            y_u      = bundle.test.y_u[mask],
            week_idx = bundle.test.week_idx[mask],
            panel_df = bundle.test.panel_df.iloc[mask] if len(bundle.test.panel_df) else bundle.test.panel_df,
        )

        # Build a temporary bundle-like object for this subset
        class _SubBundle:
            def __init__(self, b, t):
                self.test         = t
                self.n_products   = b.n_products
                self.n_retailers  = b.n_retailers
                self.schema       = b.schema
                self.name         = b.name + f"_Q{q_idx+1}"
                self.val          = b.val
                self.train        = b.train

        sub_bundle = _SubBundle(bundle, sub)

        for model_name, model in models.items():
            mr = evaluate_model_on_dataset(
                model, sub_bundle, sqp_params, true_demand_fn,
                pi_star_cache=pi_star_cache,
                cfg=cfg, verbose=False,
            )
            results[model_name][q_idx] = mr.regret_norm_mean * 100

    return results
