"""
SIDN-CRO -- Distributionally Robust Optimization demand wrapper (model B5).

Implements the mean-std uncertainty set (Gaussian DRO) as a direct comparator
for the Split Conformal Robust Optimization (CRO) used in A4.

Key methodological difference from A4 (CRO)
--------------------------------------------
B5-DRO uses the empirical standard deviation of validation residuals:

    sigma_val = std(D_hat_val - D_true_val)

and computes a robust demand estimate:

    D_tilde = max(0, D_hat - kappa * sigma_val)

where kappa is calibrated to achieve a target one-sided coverage on the
validation set.  This corresponds to the mean-std uncertainty set of
Bertsimas & Sim (2004) / Ben-Tal et al. (2009):

    U = { D : (D_hat - D) / sigma_val <= kappa }

Under Gaussian residuals, kappa = Phi^{-1}(coverage) where Phi is the
standard Normal CDF.  The key limitations of DRO vs CRO are:

  1.  It ASSUMES Gaussian (or symmetric) residuals -- a distributional
      assumption that may be violated for count-like demand data.
  2.  It provides NO finite-sample coverage guarantee: the coverage is an
      asymptotic approximation that holds only as n_cal -> inf.
  3.  It is sensitive to outliers (sigma inflated by heavy tails).

CRO (A4) is distribution-free, requires no distributional assumption, and
provides an exact finite-sample marginal coverage guarantee by design.

This module exists to provide an apples-to-apples baseline for the
experimental comparison in Table 3 / Appendix B of the manuscript.

References
----------
Ben-Tal, El Ghaoui & Nemirovski (2009) -- "Robust Optimization",
    Princeton University Press
Bertsimas & Sim (2004) -- Management Science 50(8), 1333-1346
Conforti et al. (2021) -- arXiv:2103.12393 (DRO vs CP comparison)
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Optional, Tuple

from src.models.base import DemandModel, PanelSchema


# --------------------------------------------------------------------------- #
# DRO calibration helper
# --------------------------------------------------------------------------- #
class DROCalibrator:
    """Calibrate kappa for a mean-std DRO uncertainty set.

    Uses the empirical standard deviation of calibration residuals and
    bisects (via empirical quantile) to find kappa achieving a target
    one-sided coverage.

    Parameters
    ----------
    target_coverage : desired Pr(D_true >= D_tilde) on calibration set
    """

    def __init__(self, target_coverage: float = 0.90):
        if not 0 < target_coverage < 1:
            raise ValueError(
                f"target_coverage must be in (0, 1); got {target_coverage}")
        self.target_coverage = target_coverage
        self.sigma: float    = 0.0   # std of (y_pred - y_true) on cal set
        self.kappa: float    = 0.0   # calibrated multiplier
        self.n_cal: int      = 0

    def fit(self, y_pred: np.ndarray, y_true: np.ndarray) -> "DROCalibrator":
        """Calibrate sigma and kappa on a held-out calibration set.

        Parameters
        ----------
        y_pred : (N_cal,)  model point predictions in original demand units
        y_true : (N_cal,)  true demand in original units

        Algorithm
        ---------
        1. Compute residuals r_i = y_pred_i - y_true_i  (over-prediction > 0).
        2. sigma = std(r, ddof=1).
        3. Coverage = Pr(D_true >= D_tilde) = Pr(y_true >= y_pred - k*sigma)
                    = Pr(r <= k*sigma).
           So kappa = quantile(r, target_coverage) / sigma.
        """
        n = len(y_true)
        self.n_cal = n
        residuals  = y_pred - y_true          # positive = over-prediction

        self.sigma = float(np.std(residuals, ddof=1))
        if self.sigma < 1e-8:
            # Degenerate: perfect or near-perfect predictions
            self.kappa = 0.0
            return self

        # kappa s.t. Pr(r <= kappa * sigma) = target_coverage
        target_q   = float(np.quantile(residuals, self.target_coverage))
        self.kappa = max(0.0, target_q / self.sigma)
        return self

    def robust_demand(self, y_pred: np.ndarray) -> np.ndarray:
        """Return D_tilde = max(0, y_pred - kappa * sigma)."""
        return np.maximum(0.0, y_pred - self.kappa * self.sigma)

    def coverage(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """One-sided empirical coverage: Pr(D_true >= D_tilde)."""
        D_tilde = self.robust_demand(y_pred)
        return float((y_true >= D_tilde).mean())

    def interval_width(self, y_pred: np.ndarray) -> np.ndarray:
        """Width of the implied [D_tilde, D_hat + kappa*sigma] interval
        (symmetric around D_hat)."""
        return np.full(len(y_pred), 2.0 * self.kappa * self.sigma)

    def summary(self) -> str:
        return (f"DROCalibrator(target_cov={self.target_coverage:.2f}, "
                f"n_cal={self.n_cal}, sigma={self.sigma:.3f}, "
                f"kappa={self.kappa:.3f}, "
                f"shift={self.kappa*self.sigma:.3f})")


# --------------------------------------------------------------------------- #
# B5 -- DRO demand wrapper
# --------------------------------------------------------------------------- #
class DRODemand(DemandModel):
    """B5: Mean-std Distributionally Robust Optimization demand wrapper.

    Wraps any fitted DemandModel and overrides ``raw_predict()`` to return a
    conservative demand estimate based on a Gaussian uncertainty set:

        D_tilde(x) = max(0, D_hat(x) - kappa * sigma_val)

    where sigma_val is the standard deviation of validation residuals and
    kappa is calibrated to achieve a target one-sided coverage (default 90%).

    Parameters
    ----------
    base_model       : any fitted DemandModel instance (typically B1 or B2)
    target_coverage  : desired one-sided coverage for calibration (default 0.90)

    Notes
    -----
    Unlike A4 (CRO), the coverage guarantee here is:
      - *Asymptotic* only -- not exact for finite calibration sets.
      - *Distributional* -- assumes Gaussian / symmetric residuals.
      - *Not exchangeability-based* -- no formal conformal guarantee.

    This is the standard DRO formulation and the natural alternative baseline
    to demonstrate that CRO's distribution-free guarantee has operational value
    beyond what a simpler Gaussian correction achieves.
    """

    name = "B5_DRODemand"

    def __init__(self, base_model: DemandModel,
                 target_coverage: float = 0.90):
        super().__init__(base_model.schema,
                         base_model.x_scaler,
                         base_model.y_scaler)
        self.base_model      = base_model
        self.calibrator      = DROCalibrator(target_coverage=target_coverage)
        self._calibrated     = False

    # -- DemandModel mandatory hooks ---------------------------------------- #
    def fit(self, X_cont_scaled: np.ndarray,
            X_cat: np.ndarray, y_scaled: np.ndarray, **kw) -> "DRODemand":
        """Fit the base model (delegates); call calibrate() separately."""
        self.base_model.fit(X_cont_scaled, X_cat, y_scaled, **kw)
        return self

    def raw_predict(self, X_cont_unscaled: np.ndarray,
                    X_cat: np.ndarray, **kw) -> np.ndarray:
        """Return D_tilde = max(0, D_hat - kappa * sigma_val).

        Falls back to the base model's point prediction if calibrate() has
        not been called (kappa = 0 behaviour).
        """
        D_hat = self.base_model.raw_predict(X_cont_unscaled, X_cat)
        if not self._calibrated:
            return D_hat
        return self.calibrator.robust_demand(D_hat)

    # -- Calibration -------------------------------------------------------- #
    def calibrate(self, X_cont_unscaled_cal: np.ndarray,
                  X_cat_cal: np.ndarray,
                  y_true_cal: np.ndarray) -> "DRODemand":
        """Calibrate kappa on a held-out calibration set (typically val split).

        Parameters
        ----------
        X_cont_unscaled_cal : (N_cal, n_cont)  unscaled continuous features
        X_cat_cal           : (N_cal, n_cat)
        y_true_cal          : (N_cal,)  true demand in original units

        Returns
        -------
        self (for chaining)
        """
        y_pred = self.base_model.raw_predict(X_cont_unscaled_cal, X_cat_cal)
        self.calibrator.fit(y_pred, y_true_cal)
        self._calibrated = True
        print(f"  {self.calibrator.summary()}")
        cov = self.calibrator.coverage(y_pred, y_true_cal)
        print(f"  Calibration coverage (one-sided): {cov*100:.1f}%  "
              f"(target: {self.calibrator.target_coverage*100:.0f}%)")
        return self

    # -- Diagnostics -------------------------------------------------------- #
    def predict_with_intervals(self, X_cont_unscaled: np.ndarray,
                               X_cat: np.ndarray
                               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (D_hat, D_lo, D_hi) using the symmetric Gaussian interval."""
        D_hat = self.base_model.raw_predict(X_cont_unscaled, X_cat)
        if self._calibrated:
            shift = self.calibrator.kappa * self.calibrator.sigma
            D_lo  = np.maximum(0.0, D_hat - shift)
            D_hi  = D_hat + shift
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
        dict with keys: coverage, mean_width, median_width, efficiency,
                        sigma, kappa
        """
        D_hat    = self.base_model.raw_predict(X_cont_unscaled, X_cat)
        coverage = self.calibrator.coverage(D_hat, y_true)
        widths   = self.calibrator.interval_width(D_hat)
        return {
            "coverage":      float(coverage),
            "mean_width":    float(widths.mean()),
            "median_width":  float(np.median(widths)),
            "efficiency":    float(coverage / (widths.mean() + 1e-6)),
            "sigma":         float(self.calibrator.sigma),
            "kappa":         float(self.calibrator.kappa),
        }


# --------------------------------------------------------------------------- #
# Factory: build B5 from a base model + calibration data
# --------------------------------------------------------------------------- #
def build_dro_model(
    base_model:      DemandModel,
    cal_X_cont_u:    np.ndarray,
    cal_X_cat:       np.ndarray,
    cal_y_u:         np.ndarray,
    target_coverage: float = 0.90,
) -> DRODemand:
    """Construct and calibrate a DRODemand wrapper in one call.

    Parameters
    ----------
    base_model            : fitted DemandModel (any B1-B4 recommended)
    cal_X_cont_u, cal_X_cat, cal_y_u :
        Calibration split arrays in original (unscaled) units.
        Typically the validation split from DatasetBundle.
    target_coverage       : desired one-sided coverage (default 0.90)

    Returns
    -------
    DRODemand -- calibrated and ready to use as demand function in solve_sqp()
    """
    dro = DRODemand(base_model, target_coverage=target_coverage)
    dro.calibrate(cal_X_cont_u, cal_X_cat, cal_y_u)
    return dro


# --------------------------------------------------------------------------- #
# Comparative diagnostics: CRO vs DRO side-by-side
# --------------------------------------------------------------------------- #
def compare_cro_vs_dro(
    base_model:      DemandModel,
    cal_X_cont_u:    np.ndarray,
    cal_X_cat:       np.ndarray,
    cal_y_u:         np.ndarray,
    test_X_cont_u:   np.ndarray,
    test_X_cat:      np.ndarray,
    test_y_u:        np.ndarray,
    target_coverage: float = 0.90,
) -> Dict[str, Dict[str, float]]:
    """Compare DRO vs CRO on coverage, width, and shift on the test set.

    Useful for generating the coverage comparison panel (Appendix B).

    Returns
    -------
    dict {
        "DRO": {"coverage": ..., "mean_width": ..., "sigma": ..., "kappa": ...,
                "assumption": "Gaussian (asymptotic)"},
        "CRO": {"coverage": ..., "mean_width": ..., "q_sym": ...,
                "assumption": "Distribution-free (exact finite-sample)"},
    }
    """
    from src.models.conformal import ConformalCalibrator

    # DRO
    dro       = build_dro_model(base_model, cal_X_cont_u, cal_X_cat, cal_y_u,
                                target_coverage=target_coverage)
    dro_stats = dro.evaluate_coverage(test_X_cont_u, test_X_cat, test_y_u)

    # CRO calibrated on same data, same target coverage
    alpha      = 1.0 - target_coverage
    cro_cal    = ConformalCalibrator(alpha=alpha, asymmetric=False)
    D_hat_cal  = base_model.raw_predict(cal_X_cont_u, cal_X_cat)
    cro_cal.fit(D_hat_cal, cal_y_u)
    D_hat_test = base_model.raw_predict(test_X_cont_u, test_X_cat)
    cro_cov    = cro_cal.coverage(D_hat_test, test_y_u)
    cro_width  = float(cro_cal.interval_width(D_hat_test).mean())

    return {
        "DRO": {**dro_stats,
                "assumption": "Gaussian (asymptotic)"},
        "CRO": {
            "coverage":   float(cro_cov),
            "mean_width": cro_width,
            "q_sym":      float(cro_cal.q_sym),
            "assumption": "Distribution-free (exact finite-sample)",
        },
    }
