# -*- coding: utf-8 -*-
"""Is historical advertising spend correlated with base demand over the
training weeks (within product--segment pairs)? If so, a model without time
effects attributes seasonal demand to advertising and overstates kappa."""
import json
from pathlib import Path

import numpy as np
from kt_data import build

OUT = Path(__file__).resolve().parents[1] / "results" / "known_truth" / "summary"
res = {}
for ds in ("dataco", "olist", "synth"):
    kt = build(ds, "F1", 42)
    tr = kt.bundle.train
    I, J = kt.I, kt.J
    Lw, Bw, Uw = [], [], []
    for wd in kt.train_weeks:
        r = wd.rows
        Lw.append(wd.L)
        Bw.append(sum(tr.X_cont_u[r, kt.dec[k]].reshape(I, J) for k in ("b_s", "b_m", "b_r")))
        Uw.append(np.mean([tr.X_cont_u[r, kt.dec[k]].reshape(I, J) for k in ("u_s", "u_m", "u_r")], axis=0))
    L, B, U = np.array(Lw), np.array(Bw), np.array(Uw)
    cb, cu = [], []
    for i in range(I):
        for j in range(J):
            l = np.log(np.maximum(L[:, i, j], 1e-6))
            b = np.log1p(np.maximum(B[:, i, j], 0))
            u = U[:, i, j]
            if np.std(b) > 0 and np.std(l) > 0:
                cb.append(np.corrcoef(b, l)[0, 1])
            if np.std(u) > 0 and np.std(l) > 0:
                cu.append(np.corrcoef(u, l)[0, 1])
    print(f"{ds}: within-pair corr(log(1+ad spend), log base demand) median {np.median(cb):.2f} "
          f"range [{min(cb):.2f}, {max(cb):.2f}] (n={len(cb)}); "
          f"corr(recycled share, log base demand) median {np.median(cu):.2f}; "
          f"recycled share range [{U.min():.3f}, {U.max():.3f}]")
    res[ds] = {"corr_ad_L_median": float(np.median(cb)), "corr_ad_L_min": float(min(cb)),
               "corr_ad_L_max": float(max(cb)), "corr_u_L_median": float(np.median(cu)),
               "u_min": float(U.min()), "u_max": float(U.max())}
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "confound.json").write_text(json.dumps(res, indent=1))
print("written", OUT / "confound.json")
