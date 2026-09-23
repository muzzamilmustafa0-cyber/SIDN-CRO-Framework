"""
SIDN-CRO -- Recurrent baselines (B3 LSTM, B4 Bi-LSTM + Attention).

Both models operate on per-(product, retailer) weekly sequences.  Each
sample is a sliding window of length ``T`` (default 8 weeks) ending at the
prediction week, so they capture autocorrelation in demand response to
prices, advertising, and circular-input fractions.

For consistency with the row-format DemandModel API we expose the same
``raw_predict(X_cont_unscaled, X_cat)`` signature -- internally we re-build
short context windows from the panel index stored alongside the features.

The training data must be supplied in the form produced by
``src/data_pipeline/make_train_val_test*.py`` plus a parallel ``order_meta``
matrix (n_rows, 3) with columns [product_id, retailer_id, week_idx] so we
can reconstruct sequences at training and prediction time.
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.models.base import DemandModel


# --------------------------------------------------------------------------- #
# Sequence buffer
# --------------------------------------------------------------------------- #
class _SeqBuffer:
    """Stores per-(product, retailer) chronological feature windows."""
    def __init__(self, window: int = 8):
        self.window = window
        self.store: dict[tuple[int, int], list[np.ndarray]] = {}

    def add(self, key, x_row: np.ndarray):
        self.store.setdefault(key, []).append(x_row)

    def get_window(self, key, idx: int) -> np.ndarray:
        rows = self.store[key]
        start = max(0, idx - self.window + 1)
        win = rows[start:idx + 1]
        if len(win) < self.window:
            pad = [np.zeros_like(rows[0])] * (self.window - len(win))
            win = pad + list(win)
        return np.stack(win, axis=0)


# --------------------------------------------------------------------------- #
# B3 -- 1-layer LSTM
# --------------------------------------------------------------------------- #
class _LSTM(nn.Module):
    def __init__(self, n_cont: int, n_cat: int, hidden: int = 32,
                 dropout: float = 0.10):
        super().__init__()
        in_dim = n_cont + n_cat
        self.lstm = nn.LSTM(input_size=in_dim, hidden_size=hidden,
                            batch_first=True, dropout=dropout)
        self.head = nn.Sequential(nn.Linear(hidden, hidden),
                                  nn.ReLU(),
                                  nn.Dropout(dropout),
                                  nn.Linear(hidden, 1))

    def forward(self, seq):                       # (B, T, in_dim)
        out, (h, c) = self.lstm(seq)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)


class LSTMDemand(DemandModel):
    name = "B3_LSTMDemand"

    def __init__(self, schema, x_scaler, y_scaler,
                 window: int = 8, hidden: int = 32, dropout: float = 0.10,
                 device: str = "auto"):
        super().__init__(schema, x_scaler, y_scaler)
        self.window = window
        self.device = ("cuda" if device == "auto" and torch.cuda.is_available()
                       else device if device != "auto" else "cpu")
        self.model = _LSTM(schema.n_cont, schema.n_cat,
                           hidden=hidden, dropout=dropout).to(self.device)
        self._buf = _SeqBuffer(window=window)
        self._row_idx: dict[tuple[int, int, int], int] = {}

    # ----- helper: derive (product_idx, retailer_idx) from one-hot ----------
    def _pair(self, x_cat_row):
        pcols = [i for i, c in enumerate(self.schema.cat_feats)
                 if c.startswith("product_")]
        rcols = [i for i, c in enumerate(self.schema.cat_feats)
                 if c.startswith("retailer_")]
        return (int(np.argmax(x_cat_row[pcols])),
                int(np.argmax(x_cat_row[rcols])))

    # ----- ingest training data into chronological buffers -----------------
    def _index(self, X_cont_scaled, X_cat, week_idx):
        self._buf = _SeqBuffer(window=self.window)
        self._row_idx.clear()
        order = np.argsort(week_idx)
        for new_pos, orig in enumerate(order):
            x = np.concatenate([X_cont_scaled[orig], X_cat[orig]])
            key = self._pair(X_cat[orig])
            self._buf.add(key, x)
            self._row_idx[(key[0], key[1], int(week_idx[orig]))] = (
                len(self._buf.store[key]) - 1)

    def fit(self, X_cont_scaled, X_cat, y_scaled, *,
            week_idx: np.ndarray, epochs: int = 100, batch_size: int = 32,
            lr: float = 1e-3, weight_decay: float = 1e-5,
            verbose: bool = False, seed: int = 42, **kw):
        torch.manual_seed(seed)
        self._index(X_cont_scaled, X_cat, week_idx)

        n = len(y_scaled)
        order = np.argsort(week_idx)
        seqs = np.stack([
            self._buf.get_window(self._pair(X_cat[i]),
                                 self._row_idx[(self._pair(X_cat[i])[0],
                                                self._pair(X_cat[i])[1],
                                                int(week_idx[i]))])
            for i in range(n)
        ])
        seqs_t = torch.tensor(seqs, dtype=torch.float32, device=self.device)
        y      = torch.tensor(y_scaled, dtype=torch.float32, device=self.device)

        opt = optim.Adam(self.model.parameters(), lr=lr,
                         weight_decay=weight_decay)
        loss_fn = nn.MSELoss()

        for ep in range(epochs):
            perm = torch.randperm(n, device=self.device)
            ep_loss = 0.0
            for i in range(0, n, batch_size):
                b = perm[i:i + batch_size]
                pred = self.model(seqs_t[b])
                loss = loss_fn(pred, y[b])
                opt.zero_grad()
                loss.backward()
                opt.step()
                ep_loss += loss.item() * len(b)
            if verbose and (ep + 1) % 20 == 0:
                print(f"  LSTM epoch {ep+1:3d}  loss={ep_loss / n:.4f}")
        return self

    def raw_predict(self, X_cont_unscaled, X_cat,
                    week_idx: np.ndarray | None = None):
        """For inference, the caller must pass week_idx aligned with the rows
        so we know where in the chronological history each prediction sits.
        At optimisation time (where week_idx is unknown) we fall back to the
        most recent window of the matching (product, retailer) pair."""
        scaled = self.x_scaler.transform(X_cont_unscaled).astype(np.float32)
        n = len(scaled)
        seqs = np.zeros((n, self.window, scaled.shape[1] + X_cat.shape[1]),
                        dtype=np.float32)
        for i in range(n):
            key = self._pair(X_cat[i])
            if week_idx is not None and (key[0], key[1], int(week_idx[i])) in self._row_idx:
                idx = self._row_idx[(key[0], key[1], int(week_idx[i]))]
            else:
                idx = len(self._buf.store.get(key, [])) - 1
            if idx < 0 or key not in self._buf.store:
                seqs[i, -1, :] = np.concatenate([scaled[i], X_cat[i]])
                continue
            win = self._buf.get_window(key, idx)
            # overwrite the last row with the live decision values
            win[-1] = np.concatenate([scaled[i], X_cat[i]])
            seqs[i] = win

        seqs_t = torch.tensor(seqs, dtype=torch.float32, device=self.device)
        self.model.eval()
        with torch.no_grad():
            y_scaled = self.model(seqs_t).cpu().numpy().reshape(-1, 1)
        return self.y_scaler.inverse_transform(y_scaled).ravel()


# --------------------------------------------------------------------------- #
# B4 -- Bi-LSTM with temporal attention
# --------------------------------------------------------------------------- #
class _BiLSTMAttn(nn.Module):
    def __init__(self, n_cont, n_cat, hidden=32, dropout=0.10):
        super().__init__()
        in_dim = n_cont + n_cat
        self.lstm = nn.LSTM(in_dim, hidden, batch_first=True,
                            bidirectional=True, dropout=dropout)
        self.attn = nn.Linear(2 * hidden, 1)
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden),
                                  nn.ReLU(),
                                  nn.Dropout(dropout),
                                  nn.Linear(hidden, 1))

    def forward(self, seq):
        out, _ = self.lstm(seq)                 # (B, T, 2H)
        scores = self.attn(out).squeeze(-1)      # (B, T)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
        context = (out * weights).sum(dim=1)     # (B, 2H)
        return self.head(context).squeeze(-1)


class BiLSTMAttnDemand(LSTMDemand):
    name = "B4_BiLSTMAttnDemand"

    def __init__(self, schema, x_scaler, y_scaler, *args, **kw):
        # invoke parent initialisation but swap out the inner network
        super().__init__(schema, x_scaler, y_scaler, *args, **kw)
        hidden = kw.get("hidden", 32)
        dropout = kw.get("dropout", 0.10)
        self.model = _BiLSTMAttn(schema.n_cont, schema.n_cat,
                                 hidden=hidden, dropout=dropout).to(self.device)
