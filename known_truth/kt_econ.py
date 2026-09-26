"""
Planning problem of the known-truth benchmark (pair-specific economics).

Decisions per week (I products x J segments):
    p   (I, J)     prices,                 p in [0.6 p0, 1.6 p0]
    b   (3, I, J)  advertising by echelon, b in [0, b_max]
    u   (3, I)     recycled-content share, u in [0, 0.5]

Profit (expected demand D):
    pi(z; D) = sum_ij (p_ij - c_ij(ubar_i)) D_ij - sum b - sum_i w_i sum_k u_ki^2,
    c_ij(ubar_i) = c0_ij (1 - 0.25 ubar_i)        (recycled input is cheaper)

Constraints:
    advertising budget   sum_{j,k} b_kij <= AB_i                  (linear)
    recycled input       sum_i (sum_k u_ki) D_ij <= A_j           (uses demand)
    capacity             sum_ij D_ij <= K                         (uses demand)

The vector layout used by the solver is x = [p (IJ), b_s (IJ), b_m (IJ),
b_r (IJ), u_s (I), u_m (I), u_r (I)].
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class Econ:
    I: int
    J: int
    p0: np.ndarray        # (I, J)
    c0: np.ndarray        # (I, J) unit cost at ubar = 0
    w: np.ndarray         # (I,) convex recycled-sourcing cost weight
    b_max: np.ndarray     # (3, I, J)
    AB: np.ndarray        # (I,) advertising budget per product
    A: np.ndarray         # (J,) recycled-input availability per segment
    K: float              # total capacity
    u_max: float = 0.5
    p_lo_f: float = 0.6
    p_hi_f: float = 1.6

    # ------------------------------------------------------------ layout
    @property
    def n(self):
        return 4 * self.I * self.J + 3 * self.I

    def bounds(self):
        IJ = self.I * self.J
        lo = np.concatenate([(self.p_lo_f * self.p0).ravel(), np.zeros(3 * IJ), np.zeros(3 * self.I)])
        hi = np.concatenate([(self.p_hi_f * self.p0).ravel(), self.b_max.reshape(-1),
                             np.full(3 * self.I, self.u_max)])
        return lo, hi

    def unpack(self, X):
        """X (B, n) -> p (B,I,J), b (B,3,I,J), u (B,3,I)."""
        X = np.atleast_2d(X)
        B, I, J = X.shape[0], self.I, self.J
        IJ = I * J
        p = X[:, :IJ].reshape(B, I, J)
        b = X[:, IJ:4 * IJ].reshape(B, 3, I, J)
        u = X[:, 4 * IJ:].reshape(B, 3, I)
        return p, b, u

    def pack(self, p, b, u):
        return np.concatenate([np.ravel(p), np.ravel(b), np.ravel(u)])

    # ------------------------------------------------------------ economics
    def profit(self, X, D):
        """X (B, n), D (B, I, J) expected demand -> (B,) profit."""
        p, b, u = self.unpack(X)
        ubar = u.mean(axis=1)[:, :, None]
        margin = p - self.c0 * (1.0 - 0.25 * ubar)
        return ((margin * D).sum(axis=(1, 2)) - b.sum(axis=(1, 2, 3))
                - (self.w[None, None, :] * u ** 2).sum(axis=(1, 2)))

    def margins(self, X):
        p, b, u = self.unpack(X)
        ubar = u.mean(axis=1)[:, :, None]
        return p - self.c0 * (1.0 - 0.25 * ubar)

    def constraints(self, X, D):
        """g(X) >= 0 form, (B, I + J + 1): budget (I), recycled input (J), capacity (1)."""
        p, b, u = self.unpack(X)
        budget = self.AB[None, :] - b.sum(axis=(1, 3))
        usum = u.sum(axis=1)[:, :, None]                      # (B, I, 1)
        recyc = self.A[None, :] - (usum * D).sum(axis=1)      # (B, J)
        cap = self.K - D.sum(axis=(1, 2))
        return np.concatenate([budget, recyc, cap[:, None]], axis=1)


def build_econ(p0, L_mean, b_hist, recyc_frac=0.45, cap_frac=1.0, ad_budget_mult=2.5):
    """Economics calibrated to a panel.

    p0      : (I, J) reference prices
    L_mean  : (I, J) mean weekly base demand in the training weeks
    b_hist  : (3, I, J) mean weekly historical advertising spend
    """
    I, J = p0.shape
    c0 = 0.55 * p0
    # sourcing-cost weight: the marginal cost of raising all three u_k of product i
    # to 0.5 is comparable to 60 % of the product's weekly unit-cost base
    w = 0.6 * (c0 * L_mean).sum(axis=1)
    b_max = 6.0 * np.maximum(b_hist, 1.0)
    AB = ad_budget_mult * b_hist.sum(axis=(0, 2))
    A = recyc_frac * L_mean.sum(axis=0)
    K = cap_frac * L_mean.sum()
    return Econ(I=I, J=J, p0=p0, c0=c0, w=w, b_max=b_max, AB=AB, A=A, K=K)
