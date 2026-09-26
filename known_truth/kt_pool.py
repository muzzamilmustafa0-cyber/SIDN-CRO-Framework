"""
SIDN with partial pooling of the structural parameters (SIDN-P).

Motivation (Proposition 3): the profit lost to an elasticity error grows with
its square, so noisy pair- or context-specific elasticities are costly when
historical decisions vary little. SIDN-P therefore
  1. initializes the structural heads (beta, k, alpha) at the pooled estimates
     of the classical structural model B0f (fitted on the training weeks),
  2. penalizes the context-dependent deviations from these pooled values
     (L2 penalty on the structural head weights, strength lam), and
  3. selects lam from a small grid and stops training by the loss on the
     validation weeks.
No test-week information is used. The architecture, loss, and monotonicity
guarantee are those of SIDN.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.sidn import SIDNDemand, _SIDNNet   # noqa: E402

LAM_GRID = (1e-3, 1e-2, 1e-1)


def _inv_softplus(y):
    y = max(float(y), 1e-4)
    return float(np.log(np.expm1(y)))


class SIDNPooled(SIDNDemand):
    name = "A1p_SIDNPooled"

    def _tensors(self, X_cont_scaled, X_cat, y_scaled):
        dec = self._dec_idx
        Xu = self.x_scaler.inverse_transform(X_cont_scaled).astype(np.float32)
        yu = self.y_scaler.inverse_transform(y_scaled.reshape(-1, 1)).ravel().astype(np.float32)
        t = lambda a: torch.tensor(a, dtype=torch.float32, device=self.device)
        return dict(x=t(self._make_ctx(X_cont_scaled, X_cat)), p0=t(self._p0_array(X_cat)),
                    p=t(Xu[:, dec["p"]]), bs=t(Xu[:, dec["b_s"]]), bm=t(Xu[:, dec["b_m"]]),
                    br=t(Xu[:, dec["b_r"]]), us=t(Xu[:, dec["u_s"]]), um=t(Xu[:, dec["u_m"]]),
                    ur=t(Xu[:, dec["u_r"]]), ylog=torch.log1p(torch.clamp(t(yu), min=0.0)))

    def _loss(self, T, idx=None):
        s = slice(None) if idx is None else idx
        pred = self.model(T["x"][s], T["p"][s], T["bs"][s], T["bm"][s], T["br"][s],
                          T["us"][s], T["um"][s], T["ur"][s], T["p0"][s])
        return F.huber_loss(torch.log1p(pred), T["ylog"][s], delta=1.0)

    def fit_pooled(self, train, val, pooled, lam, epochs=300, batch_size=64, lr=5e-4,
                   weight_decay=1e-5, patience=25, seed=42):
        """pooled: dict with beta, k (3,), alpha from the classical structural fit."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        dec = self._dec_idx
        Xu = self.x_scaler.inverse_transform(train.X_cont).astype(np.float32)
        keys = [self._pair(r) for r in train.X_cat]
        for key in set(keys):
            m = np.array([k == key for k in keys])
            self._p0_table[key] = float(np.median(Xu[m, dec["p"]]))
        yu = train.y_u
        init_D0 = float(np.median(yu[yu > 0])) if (yu > 0).any() else 1.0
        n_ctx = len(self._ctx_cont_idx) + self.schema.n_cat
        self.model = _SIDNNet(n_ctx, hidden=self._hidden, dropout=self._dropout,
                              init_D0=init_D0).to(self.device)
        cn = self.model.ctx_net
        with torch.no_grad():
            cn.head_beta.bias.fill_(_inv_softplus(pooled["beta"] - 0.05))
            for q in range(3):
                cn.head_k.bias[q] = _inv_softplus(pooled["k"][q])
            cn.head_alpha.bias.fill_(_inv_softplus(pooled["alpha"] - 0.10))
        Ttr = self._tensors(train.X_cont, train.X_cat, train.y)
        Tva = self._tensors(val.X_cont, val.X_cat, val.y)
        opt = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr / 20.0)
        heads = [cn.head_beta.weight, cn.head_k.weight, cn.head_alpha.weight]
        n = len(Ttr["ylog"])
        best, best_state, bad = float("inf"), None, 0
        for ep in range(epochs):
            self.model.train()
            perm = torch.randperm(n)
            for i in range(0, n, batch_size):
                b = perm[i:i + batch_size]
                loss = self._loss(Ttr, b) + lam * sum((w ** 2).sum() for w in heads)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                opt.step()
            sched.step()
            self.model.eval()
            with torch.no_grad():
                vl = float(self._loss(Tva))
            if vl < best - 1e-7:
                best, bad = vl, 0
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
        self.model.load_state_dict(best_state)
        self.val_loss = best
        return self


def fit_sidn_pooled(kt, b0f, seed):
    """Fit SIDN-P for each lam in LAM_GRID; keep the lowest validation loss."""
    b = kt.bundle
    IJ = kt.I * kt.J
    th = b0f.th
    pooled = {"beta": float(th[IJ]), "k": [float(v) for v in th[IJ + 1:IJ + 4]], "alpha": float(th[IJ + 4])}
    best = None
    for lam in LAM_GRID:
        m = SIDNPooled(b.schema, b.x_scaler, b.y_scaler, hidden=64, dropout=0.10, device="cpu")
        m.fit_pooled(b.train, b.val, pooled, lam, seed=seed)
        if best is None or m.val_loss < best[1].val_loss:
            best = (lam, m)
    best[1].lam = best[0]
    return best[1], pooled
