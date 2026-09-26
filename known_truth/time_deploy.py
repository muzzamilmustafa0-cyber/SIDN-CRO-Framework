# -*- coding: utf-8 -*-
"""Wall-clock time of the deployment steps on one CPU thread (DataCo, F1):
SIDN training, conformal calibration, and one robust weekly plan."""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kt_data import build                                             # noqa: E402
from kt_models import SrcModel, Conformal, week_rows, week_demand_fn    # noqa: E402
from kt_run import get_econ                                           # noqa: E402
from kt_solver import Planner, default_starts                         # noqa: E402
from src.models.sidn import SIDNDemand                                # noqa: E402

torch.set_num_threads(1)
out = {}
for ds in ("dataco", "olist", "synth"):
    kt = build(ds, "F1", 42)
    econ = get_econ(kt)
    b = kt.bundle
    fits = []
    for seed in (42, 43, 44):
        torch.manual_seed(seed)
        t0 = time.perf_counter()
        sidn = SIDNDemand(b.schema, b.x_scaler, b.y_scaler, hidden=64, dropout=0.10, device="cpu")
        sidn.fit(b.train.X_cont, b.train.X_cat, b.train.y, epochs=300, batch_size=64, lr=5e-4,
                 weight_decay=1e-5, patience=25, seed=seed)
        fits.append(time.perf_counter() - t0)
    m = SrcModel("A1", sidn)
    t0 = time.perf_counter()
    cf = Conformal(m, kt, 0.10)
    t_cal = time.perf_counter() - t0
    plans = []
    for wd in kt.test_weeks[:5]:
        Xc, Xk, ws = week_rows(kt, b.test, wd)
        fn = week_demand_fn(m, kt, econ, Xc, Xk, ws)
        t0 = time.perf_counter()
        Planner(econ, fn, fd_step=1e-3, obj_transform=cf.lower, con_transform=cf.upper).solve(
            default_starts(econ, kt.b_hist), max_iter=150)
        plans.append(time.perf_counter() - t0)
    out[ds] = {"n_train_rows": int(len(b.train.y)), "sidn_fit_s": fits, "calibration_s": t_cal,
               "robust_plan_s": plans}
    print(ds, f"train rows {len(b.train.y)}; SIDN fit {np.mean(fits):.1f}s (range {min(fits):.1f}-{max(fits):.1f}); "
          f"calibration {t_cal:.2f}s; robust plan {np.mean(plans):.1f}s (range {min(plans):.1f}-{max(plans):.1f})",
          flush=True)
(Path(__file__).resolve().parents[1] / "results" / "known_truth" / "summary" / "timing.json").write_text(
    json.dumps(out, indent=1))
