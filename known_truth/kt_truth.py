"""
Known-ground-truth semi-synthetic benchmark: true demand-response families.

True demand for pair (i, j) in week t at decision z = (p, b, u):

    D_ijt(z) = L_ijt * S_f(z; theta_ij, t) * exp(eps_ijt),   S_f(z_ref) = 1,

where L_ijt is the *real* base demand of the panel (qty_baseline), the response
S_f belongs to one of five families, and eps ~ N(0, sigma^2).  The families
share the same economic signs (demand falls with price and rises with
advertising and recycled content) except F5, which violates the circularity
sign for one customer segment:

    F1 matched          (p/p0)^-beta * (1 + sum_k kappa_k ln(1+b_k)) * (1 + lam*ubar)
    F2 price-misspec.   exp(-beta (p/p0 - 1)) * (same advertising and circularity terms)
    F3 saturating       (p/p0)^-beta * (1 + sum_k a_k b_k^2/(b_k^2+s_k^2))
                                     * (1 + lam * ubar^2/(ubar^2 + 0.15^2))
    F4 time-varying     F1 with beta_t = beta(1 + 0.25 sin(2 pi w/52 + phi_j)) and
                        kappa_t = kappa(1 + 0.25 cos(2 pi w/52))
    F5 sign violation   F1 with lam < 0 for the first customer segment

F1 has exactly the functional form of the SIDN structural law (the matched
case); F2-F5 are misspecified for SIDN in different ways.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

FAMILIES = ("F1", "F2", "F3", "F4", "F5")
FAMILY_NAMES = {"F1": "matched", "F2": "price-misspecified", "F3": "saturating",
                "F4": "time-varying", "F5": "sign-violating"}


@dataclass
class Truth:
    family: str
    p0: np.ndarray          # (I, J) reference price
    beta: np.ndarray        # (I, J) price elasticity at p0
    kappa: np.ndarray       # (3, I, J) advertising coefficients (s, m, r)
    lam: np.ndarray         # (I, J) circularity premium
    hill_a: np.ndarray      # (3, I, J) F3 advertising ceiling
    hill_s: np.ndarray      # (3, I, J) F3 half-saturation spend
    phase: np.ndarray       # (J,) F4 seasonal phase
    sigma: float = 0.10     # log-normal demand noise

    def response(self, p, b, u, woy):
        """S_f for a batch of decisions.

        p   : (B, I, J) prices
        b   : (B, 3, I, J) advertising (supplier, manufacturer, retailer)
        u   : (B, 3, I) recycled-content fractions
        woy : week of year (scalar) of the planning week
        returns (B, I, J)
        """
        ubar = u.mean(axis=1)[:, :, None]                       # (B, I, 1)
        rel = p / self.p0
        beta, kappa = self.beta, self.kappa
        if self.family == "F4":
            ang = 2 * np.pi * woy / 52.0
            beta = beta * (1.0 + 0.25 * np.sin(ang + self.phase))[None, :]
            kappa = kappa * (1.0 + 0.25 * np.cos(ang))
        if self.family == "F2":
            price = np.exp(-beta * (rel - 1.0))
        else:
            price = rel ** (-beta)
        if self.family == "F3":
            b2 = b ** 2
            adv = 1.0 + (self.hill_a * b2 / (b2 + self.hill_s ** 2)).sum(axis=1)
            circ = 1.0 + self.lam * ubar ** 2 / (ubar ** 2 + 0.15 ** 2)
        else:
            adv = 1.0 + (kappa * np.log1p(b)).sum(axis=1)
            circ = 1.0 + self.lam * ubar
        return price * adv * circ

    def log_elasticity(self, p, b, u, woy):
        """True price elasticity -d ln D / d ln p at the given decisions."""
        rel = p / self.p0
        beta = self.beta
        if self.family == "F4":
            ang = 2 * np.pi * woy / 52.0
            beta = beta * (1.0 + 0.25 * np.sin(ang + self.phase))[None, :]
        if self.family == "F2":
            return beta * rel
        return np.broadcast_to(beta, p.shape).copy()


def draw_truth(family: str, p0: np.ndarray, b_hist: np.ndarray, seed: int = 2026,
               sigma: float = 0.10) -> Truth:
    """Pair-heterogeneous true parameters (fixed per dataset and family).

    b_hist : (3, I, J) mean historical advertising spend per echelon and pair
    """
    I, J = p0.shape
    rng = np.random.default_rng(seed)
    beta = rng.uniform(1.6, 2.8, size=(I, J))
    kappa = np.stack([rng.uniform(0.02, 0.06, size=(I, J)),
                      rng.uniform(0.02, 0.06, size=(I, J)),
                      rng.uniform(0.03, 0.08, size=(I, J))])
    lam = rng.uniform(0.3, 1.0, size=(I, J))
    if family == "F5":
        lam[:, 0] = -rng.uniform(0.4, 0.8, size=I)
    hill_s = 2.0 * np.maximum(b_hist, 1.0)
    hill_a = 2.0 * kappa * np.log1p(hill_s)          # equals the F1 uplift at b = s
    phase = rng.uniform(0, 2 * np.pi, size=J)
    return Truth(family=family, p0=p0, beta=beta, kappa=kappa, lam=lam,
                 hill_a=hill_a, hill_s=hill_s, phase=phase, sigma=sigma)
