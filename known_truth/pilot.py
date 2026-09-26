"""Pilot: one task, with a readable summary and basic consistency checks."""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kt_run  # noqa: E402

ds, fam, seed = (sys.argv[1:4] + ["dataco", "F1", "42"][len(sys.argv[1:4]):])
kt_run.OUT.mkdir(parents=True, exist_ok=True)
t0 = time.time()
res = kt_run.run_task(ds, fam, int(seed))
print(f"{ds} {fam} seed {seed}: fit {res['time_fit_s']:.0f}s, total {res['time_total_s']:.0f}s; "
      f"q = {res['q']}, n_cal = {res['n_cal']}")
print(f"oracle: mean pi* {np.mean(res['oracle']['pi_star']):,.0f}; realised violations "
      f"{np.mean(res['oracle']['viol']):.0%}")
print(f"{'model':6} {'NR%':>7} {'viol':>6} {'cover':>6} {'el_err':>7} {'el_bias':>8} {'MAPE':>6} {'priceL%':>8} {'minNR':>7}")
for m, r in res["models"].items():
    cov = [c for c in r["cover"] if c is not None]
    print(f"{m:6} {np.mean(r['NR']):7.2f} {np.mean(r['viol']):6.0%} "
          f"{(np.mean(cov) if cov else float('nan')):6.0%} {np.mean(r['elast_err']):7.3f} "
          f"{np.mean(r['elast_bias']):8.3f} {np.mean(r['mape']):6.1f} {np.mean(r['price_loss']):8.2f} "
          f"{np.min(r['NR']):7.2f}")
(kt_run.OUT / f"pilot_{ds}_{fam}_seed{seed}.json").write_text(json.dumps(res))
