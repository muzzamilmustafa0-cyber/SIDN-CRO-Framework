"""Smoke test of kt_run_sens: one test week for one setting."""
import time
import kt_run_sens as S
t0 = time.time()
r = S.run_task("olist", "F1", 42, 0.30, 0.10, max_weeks=1)
print({m: round(r["models"][m]["NR"][0], 2) for m in S.MODELS}, "cover A4", r["models"]["A4"]["cover"],
      "fit %.0fs total %.0fs" % (r["time_fit_s"], time.time() - t0))
