"""
SIDN-CRO -- Structure-Informed Demand Network (core contribution).

The SIDN replaces the fixed-coefficient analytical demand form

    D = D0 ? (p / p0)^{-beta} ? Ad_eff(b_s, b_m, b_r; k) ? Cir_eff(u_s, u_m, u_r; alpha)

with a neural model that outputs *context-conditioned* structural coefficients:

    beta?(x),  k??(x),  k??(x),  k??(x),  alpha?(x),  D??(x)

while preserving all structural guarantees required for economic interpretability
and downstream SQP feasibility:

    dD / dp   < 0    (demand strictly decreasing in price)
    dD / db_s > 0    (advertising spend strictly increasing demand)
    dD / db_m > 0
    dD / db_r > 0
    dD / du_s > 0    (circular economy fraction raises income, increases demand)
    dD / du_m > 0
    dD / du_r > 0
    D         >= 0    (non-negative demand)

All guarantees are built directly into the network architecture via softplus
activations and the sign of each factor ? no post-hoc clipping is needed for
the monotonicity properties.

Architecture
------------
ContextNet:
    Input  : x_ctx  in R^{n_ctx}   (non-decision scaled cont features + one-hot cat)
    Trunk  : Linear -> LayerNorm -> SiLU -> Dropout -> Linear -> SiLU
    Heads  :
        log_D0  : Linear(H, 1)  ->  exp(?)           -> D?? > 0
        raw_beta   : Linear(H, 1)  ->  softplus + 0.05  -> beta?  > 0.05
        raw_k   : Linear(H, 3)  ->  softplus          -> k??,k??,k?? > 0
        raw_alpha   : Linear(H, 1)  ->  softplus + 0.10  -> alpha?  > 0.10

Structural demand formula:
    D = D??(x) ? (p / p?)^{-beta?(x)}
        ? (1 + k??(x)?ln(1+b_s) + k??(x)?ln(1+b_m) + k??(x)?ln(1+b_r))
        ? (1 + alpha?(x)?(u_s + u_m + u_r) / 3)

p? is the per-(product, retailer) median price computed from training data
(not learned), which anchors the elasticity term to an interpretable reference.

Training (MSE variant, A1):
    Loss = Huber(log1p(D?), log1p(D_true)) ? log-space Huber for lognormal demand.
    Optimiser: Adam with cosine-annealing LR and gradient clipping at norm=1.
    Early stopping with patience=20 epochs on training loss.

Decision-focused learning (DFL, A2):
    Uses forward_tensor() / make_tensors() to keep gradients alive through the
    SQP layer.  The SPO+ loss computation lives in src/train/train_sidn_dfl.py
    and calls back into these helpers.

Conformal variant (A4):
    Identical fit() call; the CRO wrapper in src/models/conformal.py adds
    prediction-interval calibration on top of raw_predict().

References
----------
Khorshidvand et al. (2021) J. Clean. Prod. -- structural demand form
Elmachtoub & Grigas (2022) Mgmt. Sci. -- SPO+ loss
Wehenkel & Louppe (2019) NeurIPS -- monotone neural networks
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, Optional, Tuple

from src.models.base import DemandModel, PanelSchema, decision_columns


# ---------------------------------------------------------------------------- #
# Internal: context -> structural-parameter network
# ---------------------------------------------------------------------------- #
class _ContextNet(nn.Module):
    """Maps non-decision context features to structural demand parameters.

    All outputs are constrained positive by construction, so demand
    monotonicity is an architectural guarantee ? not a regularisation term.

    Parameters
    ----------
    n_ctx    : width of the input context vector
    hidden   : trunk hidden width
    dropout  : dropout probability in the trunk
    init_D0  : median training demand used to bias-initialise the D? head
               so cold-start predictions are in the right order of magnitude.
    """

    def __init__(self, n_ctx: int, hidden: int = 64, dropout: float = 0.10,
                 init_D0: float = 1.0):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(n_ctx, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        # Parameter heads
        self.head_D0    = nn.Linear(hidden, 1)  # log D??  ->  exp(?) > 0
        self.head_beta  = nn.Linear(hidden, 1)  # raw      ->  softplus + 0.05 -> beta?
        self.head_k     = nn.Linear(hidden, 3)  # raw      ->  softplus -> k??,k??,k??
        self.head_alpha = nn.Linear(hidden, 1)  # raw      ->  softplus + 0.10 -> alpha?

        # Bias initialisation: cold-start parameters ? B0 defaults
        # D? head: log(median demand) so initial predictions ? median
        nn.init.constant_(self.head_D0.bias, float(np.log(max(init_D0, 1e-3))))
        nn.init.zeros_(self.head_D0.weight)

        # beta -> 0.40  (softplus(0.40) + 0.05 ? 0.47; weight=0 -> context-invariant at start)
        nn.init.constant_(self.head_beta.bias,  0.40)
        nn.init.zeros_(self.head_beta.weight)

        # k?,k?,k?  close to B0 defaults (softplus(?) ? ? for small values)
        nn.init.constant_(self.head_k.bias[0],  0.030)
        nn.init.constant_(self.head_k.bias[1],  0.030)
        nn.init.constant_(self.head_k.bias[2],  0.040)
        nn.init.zeros_(self.head_k.weight)

        # alpha_inc -> 0.80
        nn.init.constant_(self.head_alpha.bias, 0.80)
        nn.init.zeros_(self.head_alpha.weight)

    def forward(self, x_ctx: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor,
                           torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x_ctx : (B, n_ctx)

        Returns
        -------
        D0    : (B,)   baseline demand > 0
        beta  : (B,)   price elasticity > 0.05
        k     : (B, 3) advertising coefficients > 0 each
        alpha : (B,)   circularity income coefficient > 0.10
        """
        h     = self.trunk(x_ctx)
        D0    = torch.exp(self.head_D0(h)).squeeze(-1)
        beta  = F.softplus(self.head_beta(h)).squeeze(-1)  + 0.05
        k     = F.softplus(self.head_k(h))                  # (B, 3)
        alpha = F.softplus(self.head_alpha(h)).squeeze(-1) + 0.10
        return D0, beta, k, alpha


# ---------------------------------------------------------------------------- #
# Internal: full structural demand module
# ---------------------------------------------------------------------------- #
class _SIDNNet(nn.Module):
    """Assembles the structural demand formula with context-conditioned params.

    The forward pass is fully differentiable (SiLU, softplus, log, pow are all
    differentiable everywhere they are evaluated), which enables KKT-implicit
    backpropagation through the SQP layer in the DFL training variant.

    Structural demand formula (mirrors Khorshidvand et al., 2021):

        D = D??(x) ? (p / p?)^{-beta?(x)}
            ? [1 + k??(x)?ln(1+b_s) + k??(x)?ln(1+b_m) + k??(x)?ln(1+b_r)]
            ? [1 + alpha?(x)?(u_s + u_m + u_r) / 3]
    """

    def __init__(self, n_ctx: int, hidden: int = 64, dropout: float = 0.10,
                 init_D0: float = 1.0):
        super().__init__()
        self.ctx_net = _ContextNet(n_ctx, hidden=hidden,
                                   dropout=dropout, init_D0=init_D0)

    def forward(self,
                x_ctx: torch.Tensor,
                p:     torch.Tensor,
                b_s:   torch.Tensor,
                b_m:   torch.Tensor,
                b_r:   torch.Tensor,
                u_s:   torch.Tensor,
                u_m:   torch.Tensor,
                u_r:   torch.Tensor,
                p0:    torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x_ctx         : (B, n_ctx)  context (scaled non-decision cont + cat)
        p, b_s, ...     : (B,)        decision variables in **original** units
        p0            : (B,)        per-sample reference price (from lookup)

        Returns
        -------
        demand : (B,)  predicted demand in original units (>= 0)
        """
        D0, beta, k, alpha = self.ctx_net(x_ctx)
        k1, k2, k3 = k[:, 0], k[:, 1], k[:, 2]

        # -- Price elasticity: strictly decreasing in p (beta > 0, p0 > 0) ------
        p_safe  = torch.clamp(p,  min=1e-6)
        p0_safe = torch.clamp(p0, min=1e-6)
        elast   = torch.pow(p_safe / p0_safe, -beta)   # delast/dp < 0  OK

        # -- Advertising effectiveness: strictly increasing in b_s, b_m, b_r -
        b_s_c = torch.clamp(b_s, min=0.0)
        b_m_c = torch.clamp(b_m, min=0.0)
        b_r_c = torch.clamp(b_r, min=0.0)
        ad_eff = (1.0
                  + k1 * torch.log1p(b_s_c)
                  + k2 * torch.log1p(b_m_c)
                  + k3 * torch.log1p(b_r_c))           # dad_eff/db > 0 OK

        # -- Circular economy income effect: increasing in u_s, u_m, u_r ----
        u_s_c = torch.clamp(u_s, min=0.0)
        u_m_c = torch.clamp(u_m, min=0.0)
        u_r_c = torch.clamp(u_r, min=0.0)
        cir_eff = 1.0 + alpha * (u_s_c + u_m_c + u_r_c) / 3.0  # d/du > 0 OK

        # D?? > 0, elast > 0, ad_eff >= 1 > 0, cir_eff >= 1 > 0  ?  D > 0  OK
        demand = D0 * elast * ad_eff * cir_eff
        return demand   # (B,)  guaranteed positive; clamp only for safety
        # Note: DemandModel.__call__() applies np.maximum(0, ?) on the result


# ---------------------------------------------------------------------------- #
# Public: DemandModel wrapper
# ---------------------------------------------------------------------------- #
class SIDNDemand(DemandModel):
    """Structure-Informed Demand Network ? SIDN-CRO core model.

    Exposes the standard :class:`DemandModel` interface so it can replace any
    baseline in the optimisation and evaluation pipelines without code changes.

    The model can be trained with MSE (A1) or SPO+ (A2) loss; the training
    scripts in ``src/train/`` select the variant.  The ``name`` attribute is
    overwritten by the training script to reflect the variant used.

    Parameters
    ----------
    schema      : PanelSchema from the data pipeline
    x_scaler    : fitted StandardScaler for continuous features
    y_scaler    : fitted StandardScaler for the demand target
    hidden      : trunk hidden width (default 64)
    dropout     : dropout probability (default 0.10)
    device      : ``"auto"`` -> GPU if available, else CPU
    """

    name = "A1_SIDNDemand"

    def __init__(self, schema: PanelSchema, x_scaler, y_scaler,
                 hidden: int = 64, dropout: float = 0.10,
                 device: str = "auto"):
        super().__init__(schema, x_scaler, y_scaler)
        self.device = ("cuda"
                       if device == "auto" and torch.cuda.is_available()
                       else device if device != "auto" else "cpu")
        self._hidden  = hidden
        self._dropout = dropout

        # Indices of non-decision continuous features (used as context)
        dec_idx = set(decision_columns(schema).values())
        self._ctx_cont_idx = [i for i in range(schema.n_cont)
                              if i not in dec_idx]
        self._dec_idx      = decision_columns(schema)   # name -> col index
        n_ctx = len(self._ctx_cont_idx) + schema.n_cat

        # Model instantiated with placeholder init_D0; rebuilt in fit()
        self.model: _SIDNNet = _SIDNNet(
            n_ctx, hidden=hidden, dropout=dropout, init_D0=1.0
        ).to(self.device)

        # Per-(product, retailer) reference-price lookup (populated in fit())
        self._p0_table: Dict[Tuple[int, int], float] = {}

    # -- Category helpers ---------------------------------------------------- #
    def _pair(self, x_cat_row: np.ndarray) -> Tuple[int, int]:
        """Map a one-hot cat row to (product_idx, retailer_idx)."""
        pcols = [i for i, c in enumerate(self.schema.cat_feats)
                 if c.startswith("product_")]
        rcols = [i for i, c in enumerate(self.schema.cat_feats)
                 if c.startswith("retailer_")]
        return (int(np.argmax(x_cat_row[pcols])),
                int(np.argmax(x_cat_row[rcols])))

    def _p0_array(self, X_cat: np.ndarray) -> np.ndarray:
        """Build per-row reference-price vector from the lookup table."""
        return np.array(
            [self._p0_table.get(self._pair(r), 1.0) for r in X_cat],
            dtype=np.float32,
        )

    # -- Context feature construction ---------------------------------------- #
    def _make_ctx(self, X_cont_scaled: np.ndarray,
                  X_cat: np.ndarray) -> np.ndarray:
        """Concatenate non-decision scaled cont features with one-hot cat."""
        ctx_cont = X_cont_scaled[:, self._ctx_cont_idx]
        return np.concatenate([ctx_cont, X_cat], axis=1).astype(np.float32)

    # -- Fit ----------------------------------------------------------------- #
    def fit(self, X_cont_scaled: np.ndarray,
            X_cat:        np.ndarray,
            y_scaled:     np.ndarray, *,
            epochs:        int   = 200,
            batch_size:    int   = 64,
            lr:            float = 5e-4,
            weight_decay:  float = 1e-5,
            patience:      int   = 20,
            verbose:       bool  = False,
            seed:          int   = 42,
            **kw) -> "SIDNDemand":
        """Train SIDN with Huber loss on log1p-transformed demand.

        Parameters
        ----------
        X_cont_scaled : (N, n_cont)  scaled continuous features
        X_cat         : (N, n_cat)   one-hot categorical features
        y_scaled      : (N,)         scaled demand target (StandardScaler)
        epochs        : maximum training epochs
        batch_size    : mini-batch size
        lr            : initial Adam learning rate
        weight_decay  : Adam L2 regularisation
        patience      : early-stopping patience (epochs with no improvement)
        verbose       : print per-20-epoch progress
        seed          : RNG seed for reproducibility
        **kw          : absorbed (allows uniform call from train_baselines.py)
        """
        torch.manual_seed(seed)
        np.random.seed(seed)

        # -- Recover original-scale values needed for the structural formula -
        y_unscaled = self.y_scaler.inverse_transform(
            y_scaled.reshape(-1, 1)).ravel().astype(np.float32)
        X_cont_unscaled = self.x_scaler.inverse_transform(
            X_cont_scaled).astype(np.float32)

        # -- Build per-(product, retailer) reference-price table ------------
        dec  = self._dec_idx
        keys = [self._pair(X_cat[i]) for i in range(len(X_cat))]
        unique_keys = set(keys)
        for key in unique_keys:
            mask = np.array([k == key for k in keys], dtype=bool)
            self._p0_table[key] = float(
                np.median(X_cont_unscaled[mask, dec["p"]]))

        # -- Rebuild model with correct D? initialisation ------------------
        n_ctx   = len(self._ctx_cont_idx) + self.schema.n_cat
        init_D0 = float(np.median(y_unscaled[y_unscaled > 0])) \
                  if (y_unscaled > 0).any() else 1.0
        self.model = _SIDNNet(
            n_ctx, hidden=self._hidden, dropout=self._dropout,
            init_D0=init_D0
        ).to(self.device)

        # -- Move all data to device tensors --------------------------------
        def t(a: np.ndarray) -> torch.Tensor:
            return torch.tensor(a, dtype=torch.float32, device=self.device)

        x_ctx_t = t(self._make_ctx(X_cont_scaled, X_cat))
        p0_t    = t(self._p0_array(X_cat))
        p_t     = t(X_cont_unscaled[:, dec["p"]])
        b_s_t   = t(X_cont_unscaled[:, dec["b_s"]])
        b_m_t   = t(X_cont_unscaled[:, dec["b_m"]])
        b_r_t   = t(X_cont_unscaled[:, dec["b_r"]])
        u_s_t   = t(X_cont_unscaled[:, dec["u_s"]])
        u_m_t   = t(X_cont_unscaled[:, dec["u_m"]])
        u_r_t   = t(X_cont_unscaled[:, dec["u_r"]])
        y_t     = t(y_unscaled)

        # -- Optimiser + schedule -------------------------------------------
        opt   = optim.Adam(self.model.parameters(), lr=lr,
                           weight_decay=weight_decay)
        sched = optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=epochs, eta_min=lr / 20.0)

        n           = len(y_t)
        best_loss   = float("inf")
        best_state  = None
        no_improve  = 0

        y_log = torch.log1p(torch.clamp(y_t, min=0.0))  # pre-compute

        for ep in range(epochs):
            self.model.train()
            perm     = torch.randperm(n, device=self.device)
            ep_loss  = 0.0
            for i in range(0, n, batch_size):
                b    = perm[i:i + batch_size]
                pred = self.model(x_ctx_t[b], p_t[b], b_s_t[b], b_m_t[b],
                                  b_r_t[b],   u_s_t[b], u_m_t[b], u_r_t[b],
                                  p0_t[b])
                # Huber loss in log1p demand space (robust to outliers)
                loss = F.huber_loss(torch.log1p(pred), y_log[b], delta=1.0)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                opt.step()
                ep_loss += loss.item() * len(b)

            sched.step()
            ep_loss /= n

            if ep_loss < best_loss - 1e-7:
                best_loss  = ep_loss
                best_state = {k: v.clone() for k, v in
                              self.model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= patience:
                if verbose:
                    print(f"  SIDN early-stop at epoch {ep + 1:3d}  "
                          f"best_loss={best_loss:.5f}")
                break

            if verbose and (ep + 1) % 20 == 0:
                print(f"  SIDN epoch {ep+1:3d}  loss={ep_loss:.5f}")

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    # -- raw_predict --------------------------------------------------------- #
    def raw_predict(self, X_cont_unscaled: np.ndarray,
                    X_cat: np.ndarray, **kw) -> np.ndarray:
        """Return demand in **original units** for the given feature rows.

        Parameters
        ----------
        X_cont_unscaled : (N, n_cont)  unscaled continuous features
                          (decision columns already set by caller)
        X_cat           : (N, n_cat)   one-hot categorical features

        Returns
        -------
        demand : (N,)  predicted demand in original units (>= 0)
        """
        X_cont_scaled = self.x_scaler.transform(
            X_cont_unscaled).astype(np.float32)
        dec = self._dec_idx

        def t(a: np.ndarray) -> torch.Tensor:
            return torch.tensor(a.astype(np.float32),
                                dtype=torch.float32, device=self.device)

        x_ctx_t = t(self._make_ctx(X_cont_scaled, X_cat))
        p0_t    = t(self._p0_array(X_cat))
        p_t     = t(X_cont_unscaled[:, dec["p"]])
        b_s_t   = t(X_cont_unscaled[:, dec["b_s"]])
        b_m_t   = t(X_cont_unscaled[:, dec["b_m"]])
        b_r_t   = t(X_cont_unscaled[:, dec["b_r"]])
        u_s_t   = t(X_cont_unscaled[:, dec["u_s"]])
        u_m_t   = t(X_cont_unscaled[:, dec["u_m"]])
        u_r_t   = t(X_cont_unscaled[:, dec["u_r"]])

        self.model.eval()
        with torch.no_grad():
            demand = self.model(x_ctx_t, p_t, b_s_t, b_m_t, b_r_t,
                                u_s_t, u_m_t, u_r_t, p0_t).cpu().numpy()
        return np.maximum(0.0, demand.ravel())

    # -- DFL helpers (used by train_sidn_dfl.py + differentiable.py) ------- #
    def make_tensors(self, X_cont_unscaled: np.ndarray,
                     X_cat: np.ndarray) -> Dict[str, torch.Tensor]:
        """Return all input tensors for a differentiable forward pass.

        Used by the SPO+ training loop and the KKT-implicit SQP layer so that
        gradients flow back through the model parameters.

        Returns
        -------
        dict with keys: x_ctx, p0, p, b_s, b_m, b_r, u_s, u_m, u_r
        """
        X_cont_scaled = self.x_scaler.transform(
            X_cont_unscaled).astype(np.float32)
        dec = self._dec_idx

        def t(a: np.ndarray) -> torch.Tensor:
            return torch.tensor(a.astype(np.float32),
                                dtype=torch.float32, device=self.device)

        return dict(
            x_ctx = t(self._make_ctx(X_cont_scaled, X_cat)),
            p0    = t(self._p0_array(X_cat)),
            p     = t(X_cont_unscaled[:, dec["p"]]),
            b_s   = t(X_cont_unscaled[:, dec["b_s"]]),
            b_m   = t(X_cont_unscaled[:, dec["b_m"]]),
            b_r   = t(X_cont_unscaled[:, dec["b_r"]]),
            u_s   = t(X_cont_unscaled[:, dec["u_s"]]),
            u_m   = t(X_cont_unscaled[:, dec["u_m"]]),
            u_r   = t(X_cont_unscaled[:, dec["u_r"]]),
        )

    def forward_tensor(self, tensors: Dict[str, torch.Tensor]
                       ) -> torch.Tensor:
        """Differentiable forward pass ? keeps computation graph alive.

        Parameters
        ----------
        tensors : dict returned by make_tensors()

        Returns
        -------
        demand : (N,) tensor in original demand units, with gradient
        """
        return self.model(
            tensors["x_ctx"], tensors["p"],   tensors["b_s"],
            tensors["b_m"],   tensors["b_r"], tensors["u_s"],
            tensors["u_m"],   tensors["u_r"], tensors["p0"],
        )

    # -- Interpretability: extract structural parameters --------------------- #
    def inspect_params(self, X_cont_unscaled: np.ndarray,
                       X_cat: np.ndarray) -> Dict[str, np.ndarray]:
        """Return the context-conditioned structural parameters for each row.

        Useful for generating Table 3 (structural coefficient statistics) in
        the manuscript ? shows how beta, k, alpha vary across products/retailers.

        Returns
        -------
        dict with keys: D0, beta, k1, k2, k3, alpha
            each (N,) numpy array in original parameter units
        """
        X_cont_scaled = self.x_scaler.transform(
            X_cont_unscaled).astype(np.float32)
        x_ctx_t = torch.tensor(
            self._make_ctx(X_cont_scaled, X_cat),
            dtype=torch.float32, device=self.device)
        self.model.eval()
        with torch.no_grad():
            D0, beta, k, alpha = self.model.ctx_net(x_ctx_t)
        return {
            "D0":    D0.cpu().numpy(),
            "beta":  beta.cpu().numpy(),
            "k1":    k[:, 0].cpu().numpy(),
            "k2":    k[:, 1].cpu().numpy(),
            "k3":    k[:, 2].cpu().numpy(),
            "alpha": alpha.cpu().numpy(),
        }

    def summarise_params(self, X_cont_unscaled: np.ndarray,
                         X_cat: np.ndarray) -> None:
        """Print a quick summary table of structural parameters (mean +/- std).

        Mirrors the format expected in manuscript Table 3.
        """
        params = self.inspect_params(X_cont_unscaled, X_cat)
        print(f"\n{'Parameter':<12} {'Mean':>10} {'Std':>10} "
              f"{'Min':>10} {'Max':>10}")
        print("-" * 55)
        for name, vals in params.items():
            print(f"{name:<12} {vals.mean():>10.4f} {vals.std():>10.4f} "
                  f"{vals.min():>10.4f} {vals.max():>10.4f}")
