"""Smoke test of every code path of kt_run.run_task on two test weeks."""
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
import kt_run  # noqa: E402

res = kt_run.run_task(sys.argv[1] if len(sys.argv) > 1 else "olist", "F2", 43)
for m, r in res["models"].items():
    print(m, {k: (np.round(v, 3).tolist() if v and v[0] is not None and not isinstance(v[0], bool) else v)
              for k, v in r.items() if k in ("NR", "cover", "cover_hist", "dist_p", "viol", "q_path")})
print("q:", res["q"], "time", round(res["time_total_s"]))
