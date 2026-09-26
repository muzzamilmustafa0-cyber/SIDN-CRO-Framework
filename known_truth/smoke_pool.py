"""Smoke test of kt_run_pool.run_task on two test weeks (correctness only)."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kt_data  # noqa: E402

_build = kt_data.build


def short_build(*a, **k):
    kt = _build(*a, **k)
    kt.test_weeks = kt.test_weeks[:2]
    return kt


kt_data.build = short_build
import kt_run_pool  # noqa: E402

ds, fam, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
res = kt_run_pool.run_task(ds, fam, seed)
print(f"lam={res['lam']}, pooled={res['pooled']}, q={res['q']}, fit {res['time_fit_s']:.0f}s, "
      f"total {res['time_total_s']:.0f}s")
for m, r in res["models"].items():
    print(f"{m:7} NR={np.round(r['NR'], 2)} el_err={np.round(r['elast_err'], 2)} "
          f"bias={np.round(r['elast_bias'], 2)} cover={r['cover']}")
main = HERE = Path(__file__).resolve().parent.parent / "results" / "known_truth" / f"{ds}_{fam}_seed{seed}.json"
if main.exists():
    d = json.loads(main.read_text())
    print("main run, same weeks: oracle pi* =", np.round(d["oracle"]["pi_star"][:2], 1),
          "(pool run:", np.round(res["oracle"]["pi_star"], 1), ")")
    for m in ("B0f", "A1", "A4"):
        print(f"  {m:4} NR={np.round(d['models'][m]['NR'][:2], 2)} el_err={np.round(d['models'][m]['elast_err'][:2], 2)}")
