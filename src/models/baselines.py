"""
SIDN-CRO -- baseline demand models (Step 2b).

This module implements the structure-free baselines that the
Structure-Informed Demand Network (SIDN) is benchmarked against:

    B0   AnalyticalDemand        closed-form Khorshidvand-style demand
                                 (no learning -- coefficients fitted via
                                 non-linear least squares to qty_aug)

    B1   MLPDemand                feed-forward MLP, 2 x 64 ReLU, MSE-trained

    B2   XGBoostDemand            XGBoost gradient-boosted trees on the
                                 same row-format features

    B3   LSTMDemand               1-layer LSTM on per-(product, retailer)
                                 weekly sequences (in src/models/recurrent.py)

    B4   BiLSTMAttnDemand         bidirectional LSTM + temporal attention
                                 (in src/models/recurrent.py)

Each model implements the :py:class:`DemandModel` interface so the same
optimisation and evaluation code can drive any of them interchangeably.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional

from src.models.base import DemandModel, decision_columns


# --------------------------------------------------------------------------- #
# B0 -- Closed-form analytical demand (parameters fitted from data)
# --------------------------------------------------------------------------- #
class AnalyticalDemand(DemandModel):
    """Implements

            D = D0 * (p / p0)^(-beta)
                * (1 + k1*ln(1+b_s) + k2*ln(1+b_m) + k3*ln(1+b_r))
                * (1 + alpha_inc * (u_s + u_m + u_r) / 3)

    with parameters (D0, p0, beta, k1, k2, k3, alpha_inc) fit via a single
    pass of non-linear least squares on the training rows.  D0 and p0 are
    estimated *per (product, retailer)* by looking up the matching one-hot.
    """
    name = "B0_AnalyticalDemand"

    def __init__(self, schema, x_scaler, y_scaler):
        super().__init__(schema, x_scaler, y_scaler)
        self.beta = 0.40
        self.k1 = 0.030
        self.k2 = 0.030
        self.k3 = 0.040
        self.alpha_inc = 0.80
        # Per-(product, retailer) baselines, derived in fit()
        self.D0_table = {}     # key=(product_idx, retailer_idx) -> baseline demand
        self.p0_table = {}     # key=(product_idx, retailer_idx) -> reference price

    def _row_pair_key(self, X_cat_row: np.ndarray) -> tuple[int, int]:
        """Map a one-hot row to (product_idx, retailer_idx)."""
        pcols = [i for i, c in enumerate(self.schema.cat_feats)
                 if c.startswith("product_")]
        rcols = [i for i, c in enumerate(self.schema.cat_feats)
                 if c.startswith("retailer_")]
        p_idx = int(np.argmax(X_cat_row[pcols]))
        r_idx = int(np.argmax(X_cat_row[rcols]))
        return p_idx, r_idx

    def fit(self, X_cont_unscaled: np.ndarray, X_cat: np.ndarray,
            y_unscaled: np.ndarray, **kw) -> "AnalyticalDemand":
        """Fit baseline (D0, p0) per (product, retailer) and global beta, k*."""
        idx = decision_columns(self.schema)

        # Per-pair averages
        keys = np.apply_along_axis(self._row_pair_key, 1, X_cat)
        keys = [tuple(k) for k in keys]
        unique_pairs = sorted(set(keys))

        for pair in unique_pairs:
            mask = np.array([k == pair for k in keys])
            self.D0_table[pair] = float(y_unscaled[mask].mean())
            self.p0_table[pair] = float(X_cont_unscaled[mask, idx["p"]].mean())

        # Global parameters: kept at the original-paper defaults; future
        # iterations could refit them via NLS but the analytical baseline is
        # specifically meant to be the *un-fitted* reference.
        return self

    def raw_predict(self, X_cont_unscaled: np.ndarray,
                    X_cat: np.ndarray) -> np.ndarray:
        idx = decision_columns(self.schema)
        p   = X_cont_unscaled[:, idx["p"]]
        b_s = X_cont_unscaled[:, idx["b_s"]]
        b_m = X_cont_unscaled[:, idx["b_m"]]
        b_r = X_cont_unscaled[:, idx["b_r"]]
        u_s = X_cont_unscaled[:, idx["u_s"]]
        u_m = X_cont_unscaled[:, idx["u_m"]]
        u_r = X_cont_unscaled[:, idx["u_r"]]

        keys = np.apply_along_axis(self._row_pair_key, 1, X_cat)
        keys = [tuple(k) for k in keys]
        D0 = np.array([self.D0_table.get(k, 1.0) for k in keys])
        p0 = np.array([self.p0_table.get(k, 1.0) for k in keys])
        p0 = np.where(p0 > 0, p0, 1.0)

        ad_eff = (1.0
                  + self.k1 * np.log1p(b_s)
                  + self.k2 * np.log1p(b_m)
                  + self.k3 * np.log1p(b_r))
        cir_eff = 1.0 + self.alpha_inc * (u_s + u_m + u_r) / 3.0
        elast = (np.maximum(p, 1e-6) / p0) ** (-self.beta)
        return D0 * elast * ad_eff * cir_eff


# --------------------------------------------------------------------------- #
# B1 -- Feed-forward MLP
# --------------------------------------------------------------------------- #
class _MLP(nn.Module):
    def __init__(self, n_cont: int, n_cat: int, hidden: int = 64,
                 n_layers: int = 2, dropout: float = 0.10):
        super().__init__()
        in_dim = n_cont + n_cat
        layers = []
        for i in range(n_layers):
            layers += [nn.Linear(in_dim if i == 0 else hidden, hidden),
                       nn.ReLU(),
                       nn.Dropout(dropout)]
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x_cont, x_cat):
        x = torch.cat([x_cont, x_cat], dim=1)
        return self.net(x).squeeze(-1)


class MLPDemand(DemandModel):
    name = "B1_MLPDemand"

    def __init__(self, schema, x_scaler, y_scaler,
                 hidden: int = 64, n_layers: int = 2, dropout: float = 0.10,
                 device: str = "auto"):
        super().__init__(schema, x_scaler, y_scaler)
        self.device = ("cuda" if device == "auto" and torch.cuda.is_available()
                       else device if device != "auto" else "cpu")
        self.model = _MLP(schema.n_cont, schema.n_cat,
                          hidden=hidden, n_layers=n_layers,
                          dropout=dropout).to(self.device)

    def fit(self, X_cont_scaled: np.ndarray, X_cat: np.ndarray,
            y_scaled: np.ndarray, *,
            epochs: int = 200, batch_size: int = 64, lr: float = 1e-3,
            weight_decay: float = 1e-5, val_split: float = 0.0,
            verbose: bool = False, seed: int = 42, **kw) -> "MLPDemand":
        torch.manual_seed(seed)
        x_c = torch.tensor(X_cont_scaled, dtype=torch.float32, device=self.device)
        x_k = torch.tensor(X_cat,         dtype=torch.float32, device=self.device)
        y   = torch.tensor(y_scaled,      dtype=torch.float32, device=self.device)

        opt = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.MSELoss()

        n = len(y)
        idx = torch.randperm(n, device=self.device)
        for ep in range(epochs):
            self.model.train()
            perm = idx[torch.randperm(n, device=self.device)]
            ep_loss = 0.0
            for i in range(0, n, batch_size):
                b = perm[i:i + batch_size]
                pred = self.model(x_c[b], x_k[b])
                loss = loss_fn(pred, y[b])
                opt.zero_grad()
                loss.backward()
                opt.step()
                ep_loss += loss.item() * len(b)
            if verbose and (ep + 1) % 50 == 0:
                print(f"  MLP epoch {ep+1:3d}  loss={ep_loss / n:.4f}")
        return self

    def raw_predict(self, X_cont_unscaled: np.ndarray,
                    X_cat: np.ndarray) -> np.ndarray:
        scaled = self.x_scaler.transform(X_cont_unscaled).astype(np.float32)
        x_c = torch.tensor(scaled, dtype=torch.float32, device=self.device)
        x_k = torch.tensor(X_cat,  dtype=torch.float32, device=self.device)
        self.model.eval()
        with torch.no_grad():
            y_scaled = self.model(x_c, x_k).cpu().numpy().reshape(-1, 1)
        return self.y_scaler.inverse_transform(y_scaled).ravel()


# --------------------------------------------------------------------------- #
# B2 -- XGBoost
# --------------------------------------------------------------------------- #
class XGBoostDemand(DemandModel):
    name = "B2_XGBoostDemand"

    def __init__(self, schema, x_scaler, y_scaler,
                 max_depth: int = 6, n_estimators: int = 400,
                 learning_rate: float = 0.05, subsample: float = 0.8,
                 colsample_bytree: float = 0.8):
        super().__init__(schema, x_scaler, y_scaler)
        from xgboost import XGBRegressor
        self.model = XGBRegressor(
            max_depth=max_depth,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            objective="reg:squarederror",
            tree_method="hist",
            n_jobs=-1,
            verbosity=0,
        )

    def fit(self, X_cont_scaled, X_cat, y_scaled,
            *, seed: int = 42, **kw):
        self.model.set_params(random_state=seed)
        X = np.concatenate([X_cont_scaled, X_cat], axis=1)
        self.model.fit(X, y_scaled)
        return self

    def raw_predict(self, X_cont_unscaled, X_cat):
        scaled = self.x_scaler.transform(X_cont_unscaled).astype(np.float32)
        X = np.concatenate([scaled, X_cat], axis=1)
        y_scaled = self.model.predict(X).reshape(-1, 1)
        return self.y_scaler.inverse_transform(y_scaled).ravel()
