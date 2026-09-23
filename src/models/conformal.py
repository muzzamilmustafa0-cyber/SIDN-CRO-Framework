"""
SIDN-CRO -- Split Conformal Prediction wrapper (model variant A4).

This module implements the Conformal-Robust Optimization (CRO) layer that
replaces the triangular fuzzy-number / signed-distance defuzzification used
in the companion paper (Salman et al., under review).

Background: Split Conformal Prediction
---------------------------------------
Split CP (Vovk et al., 2005; Papadopoulos et al., 2002) works as follows:

    1. Train the base demand model on the training set (A1 or A2).
    2. Collect *nonconformity scores* on a held-out calibration set (here: the
       validation split):
           s_i = |D?(x_i) - D_true_i|  for each calibration row i.
    3. Compute the (1?alpha) empirical quantile of the scores:
           q?_alpha = quantile({s_i}, 1 ? alpha + 1/(n_cal + 1))
       The +1/(n_cal+1) is the finite-sample correction of Angelopoulos &
       Bates (2022) that gives *exact* marginal coverage guarantee.
    4. At test time, produce the prediction set:
           D?(x) +/- q?_alpha   (symmetric; exchangeability assumed)

Robust Optimization (CRO)
--------------------------
In the supply-chain SQP, we want decisions that are *safe* even if demand
is at the lower end of the prediction interval.  If actual demand comes in
below the model's point prediction, we over-produce, incurring waste.
Conversely, if demand exceeds the upper bound we leave money on the table.

The CRO approach (Algorithm 2 in the manuscript):
    *  Use D?_robust(x) = max(0,  D?(x) ? rho ? q?_alpha)  as the demand function
       passed to the SQP solver, where rho in [0, 1] is the robustness level.
    *  rho = 0 -> pure point prediction (no robustness)
    *  rho = 1 -> optimise against the worst-case lower bound (most conservative)

The model A4 in the paper uses rho = 0.5 as the default (50% of the conformal
radius), providing a balanced trade-off between expected profit and robustness.

Asymmetric nonconformity score (optional)
------------------------------------------
By default we use the symmetric absolute residual.  If ``asymmetric=True``
is passed to fit(), we fit two separate quantiles:

    q?_lo  = quantile(D? - D_true,   1 ? alpha/2)   (under-prediction tail)
    q?_hi  = quantile(D_true - D?,   1 ? alpha/2)   (over-prediction tail)

and produce asymmetric intervals [D? ? q?_lo, D? + q?_hi].  The robust demand
then uses only q?_lo (we hedge against under-prediction only, which causes
over-production and waste in the circular model).

References
----------
Vovk, Gammerman & Shafer (2005)   -- "Algorithmic Learning in a Random World"
Angelopoulos & Bates (2022)       -- conformal prediction tutorial, arXiv
Romano et al. (2019)              -- CQR, Conformalized Quantile Regression
Cauchois et al. (2021)            -- conformal sets for robust optimisation
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple

from src.models.base import DemandModel, PanelSchema


# --------------------------------------------------------------------------- #
# Split conformal calibration
# --------------------------------------------------------------------------- #
class ConformalCalibrator:
    """Fit and apply split conformal prediction intervals to a demand model.

    Parameters
    ----------
    alpha     : miscoverage level; intervals achieve 1?alpha coverage
    asymmetric: if True, fit separate lower / upper nonconformity quantiles
    """

    def __init__(self, alpha: float = 0.10, asymmetric: bool = False):
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0,1); got {alpha}")
        self.alpha      = alpha
        self.asymmetric = asymmetric

        # Set after fit()
        self.q_sym: float  = 0.0   # symmetric quantile (|D? - D|)
        self.q_lo:  float  = 0.0   # asymmetric: D? ? D (over-prediction errors)
        self.q_hi:  float  = 0.0   # asymmetric: D ? D? (under-prediction errors)
        self.n_cal: int    = 0

    def fit(self, y_pred: np.ndarray, y_true: np.ndarray) -> "ConformalCalibrator":
        """Calibrate on a held-out calibration set.

        Parameters
        ----------
        y_pred : (N_cal,)  model predictions in **original** demand units
        y_true : (N_cal,)  true demand in original units
        """
        n = len(y_true)
        self.n_cal = n

        # Finite-sample correction (Angelopoulos & Bates 2022, eq. 1)
        level_sym = min(1.0, (1 - self.alpha) * (1 + 1.0 / (n + 1)))

        if self.asymmetric:
            # Over-prediction tail: D? > D  ->  s_lo = D? - D > 0
            s_lo = y_pred - y_true
            # Under-prediction tail: D < D? -> s_hi = D - D? > 0
            s_hi = y_true - y_pred
            level_a = min(1.0, (1 - self.alpha / 2) * (1 + 1.0 / (n + 1)))
            self.q_lo = float(np.quantile(s_lo, level_a))
            self.q_hi = float(np.quantile(s_hi, level_a))
        else:
            scores = np.abs(y_pred - y_true)
            self.q_sym = float(np.quantile(scores, level_sym))

        return self

    def interval(self, y_pred: np.ndarray
                 ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (lower, upper) conformal prediction intervals.

        Parameters
        ----------
        y_pred : (N,)  point predictions

        Returns
        -------
        lo, hi : (N,) lower and upper bounds (lo >= 0)
        """
        if self.asymmetric:
            lo = np.maximum(0.0, y_pred - self.q_lo)
            hi = y_pred + self.q_hi
        else:
            lo = np.maximum(0.0, y_pred - self.q_sym)
            hi = y_pred + self.q_sym
        return lo, hi

    def coverage(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Empirical coverage of [lo, hi] on a test set."""
        lo, hi = self.interval(y_pred)
        return float(((y_true >= lo) & (y_true <= hi)).mean())

    def interval_width(self, y_pred: np.ndarray) -> np.ndarray:
        """Width of each prediction interval."""
        lo, hi = self.interval(y_pred)
        return hi - lo

    def summary(self) -> str:
        return (f"ConformalCalibrator(alpha={self.alpha}, "
                f"n_cal={self.n_cal}, "
                f"q_sym={self.q_sym:.3f}, "
                f"q_lo={self.q_lo:.3f}, q_hi={self.q_hi:.3f})")


# --------------------------------------------------------------------------- #
# CRO demand wrapper (model A4)
# --------------------------------------------------------------------------- #
class CRODemand(DemandModel):
    """Conformal-Robust Optimization wrapper around any base demand model.

    Wraps a fitted DemandModel (typically A1 or A2) and overrides
    ``raw_predict()`` to return a *robust* (conservative) demand estimate:

        D?_robust(x) = max(0,  D?_base(x) ? rho ? q?)

    where q? is the conformal quantile calibrated on the validation set and
    rho in [0, 1] is the robustness level.

    The SQP solver receives D?_robust as its demand function, producing
    decisions that hedge against demand shortfalls.  The coverage guarantee
    means that at the 1?alpha confidence level the true demand exceeds D?_robust,
    so the SQP solution is *feasible* under the true demand with probability >= 1?alpha.

    Parameters
    ----------
    base_model : any fitted DemandModel instance (A1 or A2 recommended)
    alpha      : miscoverage level for conformal intervals (default 0.10 -> 90% coverage)
    rho        : robustness level rho (default 0.5)
    asymmetric : use asymmetric nonconformity scores (default False)
    """

    name = "A4_CRODemand"

    def __init__(self, base_model: DemandModel,
                 alpha: float = 0.10,
                 rho:   float = 0.50,
                 asymmetric: bool = False):
        super().__init__(base_model.schema,
                         base_model.x_scaler,
                         base_model.y_scaler)
        self.base_model  = base_model
        self.rho         = rho
        self.calibrator  = ConformalCalibrator(alpha=alpha, asymmetric=asymmetric)
        self._calibrated = False

    # -- Required DemandModel hooks ------------------------------------------ #
    def fit(self, X_cont_scaled: np.ndarray,
            X_cat: np.ndarray, y_scaled: np.ndarray, **kw) -> "CRODemand":
        """Fit the base model (delegates) ? conformal calibration is separate.

        Call ``calibrate()`` after fitting to enable robust prediction.
        """
        self.base_model.fit(X_cont_scaled, X_cat, y_scaled, **kw)
        return self

    def raw_predict(self, X_cont_unscaled: np.ndarray,
                    X_cat: np.ndarray, **kw) -> np.ndarray:
        """Return robust demand: D?_base ? rho ? q?  (clipped at 0).

        If calibrate() has not been called, falls back to the base model's
        point prediction (rho = 0 behaviour).
        """
        D_hat = self.base_model.raw_predict(X_cont_unscaled, X_cat)
        if not self._calibrated:
            return D_hat
        if self.calibrator.asymmetric:
            q = self.calibrator.q_lo
        else:
            q = self.calibrator.q_sym
        D_robust = np.maximum(0.0, D_hat - self.rho * q)
        return D_robust

    # -- Conformal calibration (call after fit, on validation data) ---------- #
    def calibrate(self, X_cont_unscaled_cal: np.ndarray,
                  X_cat_cal: np.ndarray,
                  y_true_cal: np.ndarray) -> "CRODemand":
        """Calibrate conformal intervals on a held-out calibration set.

        Parameters
        ----------
        X_cont_unscaled_cal : (N_cal, n_cont)  calibration continuous features
        X_cat_cal           : (N_cal, n_cat)
        y_true_cal          : (N_cal,)  true demand in **original units**

        Returns
        -------
        self (for chaining)
        """
        y_pred = self.base_model.raw_predict(X_cont_unscaled_cal, X_cat_cal)
        self.calibrator.fit(y_pred, y_true_cal)
        self._calibrated = True
        print(f"  {self.calibrator.summary()}")
        cov = self.calibrator.coverage(y_pred, y_true_cal)
        print(f"  Calibration coverage: {cov*100:.1f}%  "
              f"(target: {(1-self.calibrator.alpha)*100:.0f}%)")
        return self

    # -- Diagnostic helpers -------------------------------------------------- #
    def predict_with_intervals(self, X_cont_unscaled: np.ndarray,
                               X_cat: np.ndarray
                               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (point prediction, lower CI, upper CI) for each row.

        Returns
        -------
        D_hat   : (N,) base model point prediction
        D_lo    : (N,) lower conformal bound
        D_hi    : (N,) upper conformal bound
        """
        D_hat = self.base_model.raw_predict(X_cont_unscaled, X_cat)
        if self._calibrated:
            D_lo, D_hi = self.calibrator.interval(D_hat)
        else:
            D_lo = D_hat.copy()
            D_hi = D_hat.copy()
        return D_hat, D_lo, D_hi

    def evaluate_coverage(self, X_cont_unscaled: np.ndarray,
                           X_cat: np.ndarray,
                           y_true: np.ndarray) -> Dict[str, float]:
        """Evaluate coverage and interval width on a test set.

        Returns
        -------
        dict with keys: coverage, mean_width, median_width, efficiency
            (efficiency = coverage / mean_width; higher is better)
        """
        D_hat = self.base_model.raw_predict(X_cont_unscaled, X_cat)
        coverage = self.calibrator.coverage(D_hat, y_true)
        widths   = self.calibrator.interval_width(D_hat)
        return {
            "coverage":      float(coverage),
            "mean_width":    float(widths.mean()),
            "median_width":  float(np.median(widths)),
            "efficiency":    float(coverage / (widths.mean() + 1e-6)),
        }


# --------------------------------------------------------------------------- #
# Factory: build A4 from A1 + calibration data
# --------------------------------------------------------------------------- #
def build_cro_model(
    base_model: DemandModel,
    cal_X_cont_u: np.ndarray,
    cal_X_cat:    np.ndarray,
    cal_y_u:      np.ndarray,
    alpha:        float = 0.10,
    rho:          float = 0.50,
    asymmetric:   bool  = False,
) -> CRODemand:
    """Construct and calibrate a CRODemand wrapper in one call.

    Parameters
    ----------
    base_model            : fitted DemandModel (A1 or A2)
    cal_X_cont_u, cal_X_cat, cal_y_u :
        Calibration split arrays in **original (unscaled)** units.
        Typically the validation split from DatasetBundle.
    alpha                 : conformal miscoverage level
    rho                   : robustness level in [0, 1]
    asymmetric            : use asymmetric nonconformity scores

    Returns
    -------
    CRODemand  ? calibrated and ready to use as the demand function in solve_sqp()
    """
    cro = CRODemand(base_model, alpha=alpha, rho=rho, asymmetric=asymmetric)
    cro.calibrate(cal_X_cont_u, cal_X_cat, cal_y_u)
    return cro


# --------------------------------------------------------------------------- #
# Robustness sensitivity: vary rho in {0, 0.25, 0.5, 0.75, 1.0}
# --------------------------------------------------------------------------- #
def rho_sensitivity(
    base_model: DemandModel,
    cal_X_cont_u: np.ndarray,
    cal_X_cat:    np.ndarray,
    cal_y_u:      np.ndarray,
    test_X_cont_u: np.ndarray,
    test_X_cat:    np.ndarray,
    test_y_u:      np.ndarray,
    alpha:         float = 0.10,
    rho_grid:      Optional[list] = None,
) -> Dict[float, Dict[str, float]]:
    """Evaluate coverage and shrinkage for different robustness levels rho.

    Returns
    -------
    dict { rho: {"coverage": ..., "mean_shrinkage": ..., "mean_width": ...} }

    Useful for generating Figure 6 (robustness sensitivity) in the manuscript.
    """
    if rho_grid is None:
        rho_grid = [0.0, 0.25, 0.50, 0.75, 1.00]

    # Calibrate once
    base_cro = CRODemand(base_model, alpha=alpha)
    base_cro.calibrate(cal_X_cont_u, cal_X_cat, cal_y_u)
    q = (base_cro.calibrator.q_lo
         if base_cro.calibrator.asymmetric
         else base_cro.calibrator.q_sym)

    D_hat = base_model.raw_predict(test_X_cont_u, test_X_cat)
    results = {}
    for rho in rho_grid:
        D_robust  = np.maximum(0.0, D_hat - rho * q)
        shrinkage = float((D_hat - D_robust).mean())
        coverage  = float(((test_y_u >= D_robust)).mean())  # one-sided
        width     = rho * q
        results[rho] = {
            "coverage":       coverage,
            "mean_shrinkage": shrinkage,
            "width":          float(width),
        }
    return results
