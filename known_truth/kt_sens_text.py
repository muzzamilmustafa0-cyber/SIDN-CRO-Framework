# -*- coding: utf-8 -*-
"""
Sensitivity analysis of the known-truth benchmark (runs of kt_run_sens.py).

Historical price variation sigma_p in {0.05, 0.15, 0.30} (families F1, F2) and
demand noise sigma in {0.05, 0.10, 0.20} (family F1); the middle values are the
base setting of the main runs, restricted to the same seeds and families. Writes
body_sens.tex (a Results subsection with its figure), figs/fig_kt_sens.png, and
sens_numbers.json into KT_PAPER_DIR. Every statement is checked (FLAG if not).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams["pdf.fonttype"] = 42  # embed TrueType fonts in PDF figures
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results" / "known_truth"
SENS = HERE.parent / "results" / "known_truth_sens"
V2 = Path(os.environ.get("KT_PAPER_DIR", str(HERE.parent / "results" / "paper")))
(V2 / "figs").mkdir(parents=True, exist_ok=True)   # output folder of the tables, figures, and text
MODELS = ["B0f", "B2m", "A1", "A4", "B2c"]
SEEDS = [42, 43, 44]
flags = []
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.spines.top": False,
                     "axes.spines.right": False})


def check(c, msg):
    if not c:
        flags.append(msg)
    return c


def rows_from(f, sp, sd):
    d = json.loads(f.read_text())
    if "models" not in d:
        return []
    out = []
    for m in MODELS:
        if m not in d["models"]:
            continue
        r = d["models"][m]
        cov = [c for c in r["cover"] if c is not None]
        covh = [c for c in r.get("cover_hist", []) if c is not None]
        out.append({"dataset": d["dataset"], "family": d["family"], "seed": d["seed"], "model": m,
                    "sigma_p": sp, "sigma": sd, "NR": np.mean(r["NR"]), "viol": np.mean(r["viol"]),
                    "cover": np.mean(cov) if cov else np.nan, "cover_hist": np.mean(covh) if covh else np.nan,
                    "elast_err": np.mean(r["elast_err"]), "price_loss": np.mean(r["price_loss"])})
    return out


def load():
    rows = []
    for tag, (sp, sd) in {"sp0.05": (0.05, 0.10), "sp0.30": (0.30, 0.10),
                          "sd0.05": (0.15, 0.05), "sd0.20": (0.15, 0.20)}.items():
        for f in sorted((SENS / tag).glob("*_seed*.json")):
            rows += rows_from(f, sp, sd)
    for f in sorted(RES.glob("*_seed*.json")):          # base setting (main runs)
        if f.name.startswith("pilot"):
            continue
        d = json.loads(f.read_text())
        if "models" in d and d["seed"] in SEEDS and d["family"] in ("F1", "F2"):
            rows += rows_from(f, 0.15, 0.10)
    return pd.DataFrame(rows)


def balanced(df, by):
    """Equal weight per panel--family cell."""
    return df.groupby(by + ["dataset", "family"]).mean(numeric_only=True).groupby(level=by).mean()


def main():
    df = load()
    # only runs complete in every setting of a comparison are used (matched seeds and cells)
    key = ["dataset", "family", "seed"]
    sp = df[(df.sigma == 0.10) & df.family.isin(["F1", "F2"])]
    sp_keys = sp.groupby(key).sigma_p.nunique()
    sp = sp.set_index(key).loc[sp_keys[sp_keys == 3].index].reset_index()
    sd = df[(df.sigma_p == 0.15) & (df.family == "F1")]
    sd_keys = sd.groupby(key).sigma.nunique()
    sd = sd.set_index(key).loc[sd_keys[sd_keys == 3].index].reset_index()
    n_sp, n_sd = int((sp_keys == 3).sum()), int((sd_keys == 3).sum())
    print(f"matched runs: price variation {n_sp}, noise {n_sd}")
    if n_sp < 6 or n_sd < 3:
        print("too few matched runs for the sensitivity analysis")
        return

    used = sorted(set(sp.seed) | set(sd.seed))
    seed_txt = (f"seeds {used[0]}--{used[-1]}" if len(used) > 2 and used == list(range(used[0], used[-1] + 1))
                else ("seeds " + " and ".join(map(str, used)) if len(used) > 1 else f"seed {used[0]}"))
    A = balanced(sp, ["sigma_p", "model"])
    Bn = balanced(sd, ["sigma", "model"])
    g = lambda T, lev, m, col: float(T.loc[(lev, m), col])
    S = {"n_sp": n_sp, "n_sd": n_sd, "seeds_used": [int(s) for s in used], "seed_txt": seed_txt}
    for m in ("A1", "B0f", "B2m", "A4", "B2c"):
        for lev in (0.05, 0.15, 0.30):
            S[f"sp{lev}_{m}"] = {c: g(A, lev, m, c) for c in ("NR", "elast_err", "cover", "viol")}
        for lev in (0.05, 0.10, 0.20):
            S[f"sd{lev}_{m}"] = {c: g(Bn, lev, m, c) for c in ("NR", "elast_err", "cover", "cover_hist", "viol")}

    def mono_dec(m, col, pref="sp", levs=(0.05, 0.15, 0.30)):
        v = [S[f"{pref}{l}_{m}"][col] for l in levs]
        return v[0] > v[1] > v[2]

    # ---- claims, each checked
    e_dec = {m: mono_dec(m, "elast_err") for m in ("A1", "B0f", "B2m")}
    r_dec = {m: mono_dec(m, "NR") for m in ("A1", "B0f")}
    gap_lo = S["sp0.05_B2m"]["NR"] - max(S["sp0.05_A1"]["NR"], S["sp0.05_B0f"]["NR"])
    gap_hi = S["sp0.3_B2m"]["NR"] - max(S["sp0.3_A1"]["NR"], S["sp0.3_B0f"]["NR"])
    check(gap_lo > 0 and gap_hi > 0, "sensitivity: structure not ahead of black box at every price variation")
    a1_vs_b0 = {l: S[f"sp{l}_A1"]["NR"] - S[f"sp{l}_B0f"]["NR"] for l in (0.05, 0.15, 0.3)}
    cov_x = [S[f"sd{l}_B2c"]["cover"] for l in (0.05, 0.10, 0.20)]
    cov_s = [S[f"sd{l}_A4"]["cover"] for l in (0.05, 0.10, 0.20)]
    check(all(s > x for s, x in zip(cov_s, cov_x)), "sensitivity: SIDN band not above black-box band at every noise level")
    # the two main differences in the other dimension: coverage across price variation, regret across noise
    cov_s_sp = [S[f"sp{l}_A4"]["cover"] for l in (0.05, 0.15, 0.3)]
    cov_x_sp = [S[f"sp{l}_B2c"]["cover"] for l in (0.05, 0.15, 0.3)]
    nr_noise = {m: [S[f"sd{l}_{m}"]["NR"] for l in (0.05, 0.10, 0.20)] for m in ("A1", "B0f", "B2m")}
    cov_gap_sp = all(s > x for s, x in zip(cov_s_sp, cov_x_sp))
    reg_gap_noise = all(nr_noise["B2m"][k] > max(nr_noise["A1"][k], nr_noise["B0f"][k]) for k in range(3))
    check(cov_gap_sp, "sensitivity: SIDN band not above black-box band at every price variation")
    check(reg_gap_noise, "sensitivity: structure not ahead of black box at every noise level")
    x_rises = cov_x_sp[0] < cov_x_sp[1] < cov_x_sp[2]
    nr4 = [S[f"sd{l}_A4"]["NR"] for l in (0.05, 0.10, 0.20)]
    nr1 = [S[f"sd{l}_A1"]["NR"] for l in (0.05, 0.10, 0.20)]
    price_of_guar = [a - b for a, b in zip(nr4, nr1)]

    def f1(x):
        return f"{x:.1f}"

    def pc(x):
        return f"{100 * x:.0f}"

    sens_elast = ("The elasticity error of both structural models falls as the historical prices vary more"
                  if e_dec["A1"] and e_dec["B0f"] else
                  "The elasticity error of the structural models does not fall monotonically with the historical price variation")

    def trend(m):
        v = [S[f"sp{l}_{m}"]["NR"] for l in (0.05, 0.15, 0.3)]
        if v[0] > v[1] > v[2]:
            return f"falls from {f1(v[0])}\\% to {f1(v[2])}\\%"
        if max(v) - min(v) < 1.0:
            return f"stays at about {f1(float(np.mean(v)))}\\%"
        if v[2] < v[0]:
            return f"falls from {f1(v[0])}\\% to {f1(v[2])}\\%, though not monotonically"
        return f"rises from {f1(v[0])}\\% to {f1(v[2])}\\%"
    d = [a1_vs_b0[l] for l in (0.05, 0.15, 0.3)]
    if d[0] > d[1] > d[2]:
        adv = (f"The advantage of \\SIDN over the classical model therefore grows with the price variation "
               f"(difference ${d[0]:+.1f}$, ${d[1]:+.1f}$, and ${d[2]:+.1f}$ points), consistent with the view that "
               f"context-dependent elasticities need identifying variation.")
    else:
        adv = f"The difference between \\SIDN and the classical model is ${d[0]:+.1f}$, ${d[1]:+.1f}$, and ${d[2]:+.1f}$ points."
    sens_reg = f"The regret of \\SIDN {trend('A1')}, whereas that of the classical model {trend('B0f')}. {adv}"
    cov_sp_sentence = ""
    if cov_gap_sp:
        cov_sp_sentence = (
            f" At the planned decisions, the band of \\SIDNCRO covers the realized demand in "
            f"{pc(cov_s_sp[0])}\\%, {pc(cov_s_sp[1])}\\%, and {pc(cov_s_sp[2])}\\% of the weeks at the three levels of price "
            f"variation, against {pc(cov_x_sp[0])}\\%, {pc(cov_x_sp[1])}\\%, and {pc(cov_x_sp[2])}\\% for the black-box band"
            + (", whose coverage rises with the historical price variation, consistent with "
               "Theorem~\\ref{thm:transport}: the wider the historical decisions, the less of the plan lies "
               "outside them." if x_rises else "."))
    text = rf"""
\subsection{{Sensitivity to Historical Price Variation and Demand Noise}}
\label{{ssec:res_sens}}

Two parameters of the benchmark bear directly on the theory: the variation of
the historical prices, from which elasticities must be identified
(Proposition~\ref{{prop:elasticity}}), and the demand noise, which sets the width of the
conformal band. We varied each around the base setting, the price variation
$\sigma_p\in\{{0.05,0.15,0.30\}}$ in families F1 and F2 and the noise
$\sigma\in\{{0.05,0.10,0.20\}}$ in F1, for {seed_txt} and all three panels, with
the models, limits, and planner of the main runs ({n_sp} and {n_sd} matched runs;
Fig.~\ref{{fig:sens}}). {sens_elast} (\SIDN from {S['sp0.05_A1']['elast_err']:.2f} to
{S['sp0.3_A1']['elast_err']:.2f}, classical model from {S['sp0.05_B0f']['elast_err']:.2f} to
{S['sp0.3_B0f']['elast_err']:.2f}). {sens_reg} The best black-box model, monotone XGBoost, forfeits
{f1(S['sp0.05_B2m']['NR'])}\%, {f1(S['sp0.15_B2m']['NR'])}\%, and {f1(S['sp0.3_B2m']['NR'])}\% at the three levels, so the
advantage of structure persists at every level of price variation.{cov_sp_sentence} Across noise levels, the band of \SIDNCRO covers the
realized demand at the planned decisions in {pc(cov_s[0])}\%, {pc(cov_s[1])}\%, and {pc(cov_s[2])}\% of the
weeks, against {pc(cov_x[0])}\%, {pc(cov_x[1])}\%, and {pc(cov_x[2])}\% for the black-box band, and the
price of the guarantee (regret of \SIDNCRO minus that of \SIDN) is {f1(price_of_guar[0])},
{f1(price_of_guar[1])}, and {f1(price_of_guar[2])} points.

\begin{{figure*}}[!t]
\centering
\includegraphics[width=\linewidth]{{figs/fig_kt_sens}}
\caption{{Sensitivity analysis ({seed_txt}, all panels; equal weight per
panel--family cell). (a) Normalized regret and (b) mean absolute elasticity
error at the planned decisions against the historical price variation
$\sigma_p$ (families F1 and F2). (c) Coverage of the conformal band at the
planned decisions against the demand noise $\sigma$ (family F1). The middle
values are the base setting of the main experiments.}}
\label{{fig:sens}}
\end{{figure*}}
"""
    # ---- figure: legends above the panels, never inside the plotting area
    from matplotlib.lines import Line2D
    fig = plt.figure(figsize=(7.0, 2.75))
    gs = fig.add_gridspec(1, 3, wspace=0.42)
    ax = [fig.add_subplot(gs[k]) for k in range(3)]
    cols = {"A1": ("#d94801", "o", "SIDN (A1)"), "B0f": ("#525252", "s", "Classical (B0f)"),
            "B2m": ("#08519c", "^", "Monotone XGBoost (B2m)")}
    levs = [0.05, 0.15, 0.30]
    key_ = lambda l: 0.3 if l == 0.30 else l
    for m, (c, mk, lab) in cols.items():
        ax[0].plot(levs, [S[f"sp{key_(l)}_{m}"]["NR"] for l in levs], marker=mk, color=c, lw=1.3, ms=4)
        ax[1].plot(levs, [S[f"sp{key_(l)}_{m}"]["elast_err"] for l in levs], marker=mk, color=c, lw=1.3, ms=4)
    nl = [0.05, 0.10, 0.20]
    ax[2].plot(nl, [100 * v for v in cov_s], marker="o", color="#d94801", lw=1.3, ms=4)
    ax[2].plot(nl, [100 * v for v in cov_x], marker="D", color="#2171b5", lw=1.3, ms=3.6)
    ax[2].axhline(90, color="0.45", lw=0.9, ls="--")
    ax[2].set_ylim(-5, 105)
    ax[0].set_ylim(bottom=0)
    ax[1].set_ylim(bottom=0)
    for a_, xl, yl, tl in ((ax[0], r"Historical price variation $\sigma_p$", "Normalized regret (%)", "(a) Regret"),
                           (ax[1], r"Historical price variation $\sigma_p$", "Elasticity error", "(b) Elasticity error"),
                           (ax[2], r"Demand noise $\sigma$", "Coverage at plan (%)", "(c) Coverage at plan")):
        a_.set_xlabel(xl)
        a_.set_ylabel(yl)
        a_.set_title(tl, loc="left", fontsize=8.5)
        a_.tick_params(labelsize=7.5)
    ax[0].set_xticks(levs)
    ax[1].set_xticks(levs)
    ax[2].set_xticks(nl)
    # one legend above all panels: first row for panels (a) and (b), second row for panel (c)
    # (entries are filled column by column, hence the interleaved order)
    hm = {m: Line2D([], [], color=c, marker=mk, lw=1.3, ms=4) for m, (c, mk, _) in cols.items()}
    ha4 = Line2D([], [], color="#d94801", marker="o", lw=1.3, ms=4)
    hb2c = Line2D([], [], color="#2171b5", marker="D", lw=1.3, ms=3.6)
    hnom = Line2D([], [], color="0.45", lw=0.9, ls="--")
    fig.legend([hm["A1"], ha4, hm["B0f"], hb2c, hm["B2m"], hnom],
               ["(a, b) SIDN (A1)", "(c) SIDN-CRO (A4)", "(a, b) Classical (B0f)", "(c) XGBoost + CRO (B2c)",
                "(a, b) Monotone XGBoost (B2m)", "(c) Nominal coverage 90%"],
               loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=7,
               columnspacing=1.6, handletextpad=0.4)
    fig.subplots_adjust(top=0.72, bottom=0.17, left=0.07, right=0.99)
    (V2 / "figs").mkdir(exist_ok=True)
    fig.savefig(V2 / "figs" / "fig_kt_sens.png", dpi=300, bbox_inches="tight")
    fig.savefig(V2 / "figs" / "fig_kt_sens.pdf", bbox_inches="tight")  # vector copy for the journal artwork
    plt.close(fig)
    (V2 / "body_sens.tex").write_text(text, encoding="utf-8")
    S.update({"cov_by_sp": {"A4": cov_s_sp, "B2c": cov_x_sp}, "nr_by_noise": nr_noise,
              "cov_gap_all_sp": cov_gap_sp, "regret_gap_all_noise": reg_gap_noise})
    S.update({"elast_decreasing": e_dec, "regret_decreasing": r_dec, "a1_minus_b0f": a1_vs_b0,
              "cov_A4": cov_s, "cov_B2c": cov_x, "price_of_guarantee": price_of_guar})
    (V2 / "sens_numbers.json").write_text(json.dumps(S, indent=1), encoding="utf-8")
    print("body_sens.tex written")
    print("FLAGS:" if flags else "no flags", *flags, sep="\n  ")


if __name__ == "__main__":
    main()
