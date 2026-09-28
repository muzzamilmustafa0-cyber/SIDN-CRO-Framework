# -*- coding: utf-8 -*-
"""
Mechanism figure: the demand response and conformal band that XGBoost and SIDN
learn for one product--segment pair, against the true response, inside and
outside the historical price range, with the prices that the robust planners
and the oracle choose. One benchmark instance (panel, truth family F1, seed 42),
first test week, the pair with the largest base demand; all other decisions are
held at the week's historical values.

Writes figs/fig_kt_mechanism.png into KT_PAPER_DIR and prints the numbers used
in the caption and text (also saved as mechanism.json).
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import matplotlib
matplotlib.rcParams["pdf.fonttype"] = 42  # embed TrueType fonts in PDF figures
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kt_data import build                                        # noqa: E402
from kt_models import Conformal, week_rows, week_demand_fn        # noqa: E402
from kt_run import get_econ                                      # noqa: E402
from kt_run_sens import fit_subset                               # noqa: E402
from kt_solver import Planner, default_starts                    # noqa: E402

V2 = Path(os.environ.get("KT_PAPER_DIR", str(Path(__file__).resolve().parents[1] / "results" / "paper")))
(V2 / "figs").mkdir(parents=True, exist_ok=True)   # output folder of the tables, figures, and text
DS = sys.argv[1] if len(sys.argv) > 1 else "dataco"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.spines.top": False,
                     "axes.spines.right": False})

kt = build(DS, "F1", 42)
econ = get_econ(kt)
models = fit_subset(kt, 42)
cf_s = Conformal(models["A1"], kt, 0.10)
cf_x = Conformal(models["B2"], kt, 0.10)
I, J = kt.I, kt.J
wd = kt.test_weeks[0]
test = kt.bundle.test
Xc, Xk, ws = week_rows(kt, test, wd)
i, j = np.unravel_index(np.argmax(wd.L), wd.L.shape)
p_h = Xc[:, kt.dec["p"]].reshape(I, J)
b_h = np.stack([Xc[:, kt.dec[k]].reshape(I, J) for k in ("b_s", "b_m", "b_r")])
u_h = np.stack([Xc[:, kt.dec[k]].reshape(I, J)[:, 0] for k in ("u_s", "u_m", "u_r")])
x_h = econ.pack(p_h, b_h, u_h)
p0 = kt.p0[i, j]
grid = np.linspace(0.6, 1.6, 101) * p0
X = np.repeat(x_h[None], len(grid), axis=0)
X[:, i * J + j] = grid

tru = kt.truth
true_fn = lambda Z: wd.L * tru.response(*econ.unpack(Z), wd.woy) * np.exp(tru.sigma ** 2 / 2)
D_true = true_fn(X)[:, i, j]
f_s = week_demand_fn(models["A1"], kt, econ, Xc, Xk, ws)
f_x = week_demand_fn(models["B2"], kt, econ, Xc, Xk, ws)
Ds, Dx = f_s(X), f_x(X)
lo_s, hi_s = cf_s.lower(Ds)[:, i, j], cf_s.upper(Ds)[:, i, j]
lo_x, hi_x = cf_x.lower(Dx)[:, i, j], cf_x.upper(Dx)[:, i, j]
Ds, Dx = Ds[:, i, j], Dx[:, i, j]


def elast(D):
    return -np.gradient(np.log(np.maximum(D, 1e-9)), np.log(grid))


# historical prices of this pair over the training weeks
tr = kt.bundle.train
hist_p = np.array([tr.X_cont_u[w.rows, kt.dec["p"]].reshape(I, J)[i, j] for w in kt.train_weeks])
# plans of the week: oracle, SIDN-CRO, XGBoost + CRO
orc = Planner(econ, true_fn, fd_step=1e-5).solve(default_starts(econ, kt.b_hist, extra=3, seed=42))
pl_s = Planner(econ, f_s, fd_step=1e-3, obj_transform=cf_s.lower, con_transform=cf_s.upper).solve(
    default_starts(econ, kt.b_hist), max_iter=150)
pl_x = Planner(econ, f_x, fd_step=1e-3, obj_transform=cf_x.lower, con_transform=cf_x.upper).solve(
    default_starts(econ, kt.b_hist), max_iter=150)
pr = {k: float(econ.unpack(v["x"][None])[0][0, i, j] / p0) for k, v in
      (("oracle", orc), ("SIDN-CRO", pl_s), ("XGBoost+CRO", pl_x))}


def at(rel, arr):
    return float(np.interp(rel * p0, grid, arr))


facts = {"panel": DS, "pair": [int(i), int(j)], "hist_rel_range": [float(hist_p.min() / p0), float(hist_p.max() / p0)],
         "planned_rel_price": pr, "true_elasticity": float(np.median(elast(D_true))),
         "sidn_elast_at_plan": at(pr["SIDN-CRO"], elast(Ds)), "xgb_elast_at_plan": at(pr["XGBoost+CRO"], elast(Dx)),
         "true_D_at_xgb_plan": at(pr["XGBoost+CRO"], D_true),
         "xgb_band_at_xgb_plan": [at(pr["XGBoost+CRO"], lo_x), at(pr["XGBoost+CRO"], hi_x)],
         "true_D_at_sidn_plan": at(pr["SIDN-CRO"], D_true),
         "sidn_band_at_sidn_plan": [at(pr["SIDN-CRO"], lo_s), at(pr["SIDN-CRO"], hi_s)]}
print(json.dumps(facts, indent=1))


# ---------------------------------------------------------------- arc elasticities (5% price steps)
STEP = 1.05
X2 = X.copy()
X2[:, i * J + j] = grid * STEP
Dt2 = true_fn(X2)[:, i, j]
Ds2, Dx2 = f_s(X2)[:, i, j], f_x(X2)[:, i, j]


def arc(D1, D2):
    return -(np.log(np.maximum(D2, 1e-9)) - np.log(np.maximum(D1, 1e-9))) / np.log(STEP)


e_t, e_s, e_x = arc(D_true, Dt2), arc(Ds, Ds2), arc(Dx, Dx2)
facts.update({"elasticity_measure": "arc elasticity over 5% price increases",
              "true_elasticity": float(np.median(e_t)),
              "sidn_elast_at_plan": at(pr["SIDN-CRO"], e_s),
              "xgb_elast_at_plan": at(pr["XGBoost+CRO"], e_x)})
print("arc elasticities at the plans:", facts["true_elasticity"], facts["sidn_elast_at_plan"], facts["xgb_elast_at_plan"])

# ---------------------------------------------------------------- figure
from matplotlib.lines import Line2D                    # noqa: E402
from matplotlib.patches import Patch                   # noqa: E402
from matplotlib.legend_handler import HandlerTuple     # noqa: E402

BLUE, ORANGE, SHADE = "#2171b5", "#d94801", "0.90"
fig = plt.figure(figsize=(7.0, 3.1))
gs = fig.add_gridspec(1, 2, wspace=0.28)
a1, a2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
r = grid / p0
hr = (hist_p.min() / p0, hist_p.max() / p0)
for a in (a1, a2):
    a.axvspan(*hr, color=SHADE, zorder=0, lw=0)
a1.fill_between(r, lo_x, hi_x, color=BLUE, alpha=0.16, lw=0, zorder=1)
a1.fill_between(r, lo_s, hi_s, color=ORANGE, alpha=0.16, lw=0, zorder=1)
a1.plot(r, D_true, color="black", lw=1.6, zorder=4)
a1.plot(r, Dx, color=BLUE, lw=1.4, zorder=3)
a1.plot(r, Ds, color=ORANGE, lw=1.4, zorder=3)
a2.plot(r, e_t, color="black", lw=1.6, zorder=4)
a2.plot(r, e_x, color=BLUE, lw=1.4, zorder=3)
a2.plot(r, e_s, color=ORANGE, lw=1.4, zorder=3)
plans = (("oracle", "black", ":"), ("SIDN-CRO", ORANGE, "--"), ("XGBoost+CRO", BLUE, "--"))
for a in (a1, a2):
    for k, c, ls in plans:
        a.axvline(pr[k], color=c, ls=ls, lw=1.1, zorder=2)
    a.set_xlim(0.58, 1.62)
    a.set_xlabel(r"Price relative to reference price, $p/p_0$")
    a.tick_params(labelsize=7.5)
a1.set_ylim(0, 1.04 * max(D_true.max(), hi_s.max(), hi_x.max()))
a1.set_ylabel("Weekly demand (units)")
hi_e = max(e_t.max(), e_s.max(), e_x.max())
FLOOR = -3.0
a2.set_ylim(FLOOR, hi_e + 0.4)
a2.set_ylabel("Price elasticity (5% arc)")
if e_x.min() < FLOOR:
    k = int(np.argmin(e_x))
    # short label in the empty area left of the dip; the explanation is in the caption
    a2.text(r[k] + 0.05, FLOOR + 0.25, f"min {e_x.min():.0f}".replace("-", "−"), fontsize=6.5,
            color=BLUE, ha="left", va="bottom")
    facts["xgb_min_arc_elasticity"] = [float(e_x.min()), float(r[k])]
a1.set_title("(a) Demand and 90% conformal band", loc="left", fontsize=8.5)
a2.set_title("(b) Price elasticity", loc="left", fontsize=8.5)

handles = [Line2D([], [], color="black", lw=1.6),
           (Patch(facecolor=BLUE, alpha=0.16, lw=0), Line2D([], [], color=BLUE, lw=1.4)),
           (Patch(facecolor=ORANGE, alpha=0.16, lw=0), Line2D([], [], color=ORANGE, lw=1.4)),
           Patch(facecolor=SHADE, lw=0),
           Line2D([], [], color="black", ls=":", lw=1.1),
           Line2D([], [], color=ORANGE, ls="--", lw=1.1),
           Line2D([], [], color=BLUE, ls="--", lw=1.1)]
labels = ["True expected demand", "XGBoost and its band (B2c)", "SIDN and its band (SIDN-CRO)",
          "Historical price range", "Oracle price", "SIDN-CRO price", "XGBoost + CRO price"]
fig.legend(handles, labels, handler_map={tuple: HandlerTuple(ndivide=None, pad=0)}, loc="upper center",
           bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False, fontsize=7.2, columnspacing=1.4,
           handlelength=2.4, handletextpad=0.5)
fig.subplots_adjust(top=0.74, bottom=0.15, left=0.075, right=0.99)
(V2 / "figs").mkdir(exist_ok=True)
fig.savefig(V2 / "figs" / "fig_kt_mechanism.png", dpi=300, bbox_inches="tight")
fig.savefig(V2 / "figs" / "fig_kt_mechanism.pdf", bbox_inches="tight")  # vector copy for the journal artwork
(Path(__file__).resolve().parents[1] / "results" / "known_truth" / "summary" / "mechanism.json").write_text(
    json.dumps(facts, indent=1))
print("figure written")
