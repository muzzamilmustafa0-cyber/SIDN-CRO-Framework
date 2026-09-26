"""
LaTeX tables and figures for the manuscript, generated from the benchmark results
(no hand-copied numbers). Writes into Mathematics Paper/IEEE_TEM_v2/:
    tab_main.tex          normalized regret by panel x family x model (mean over seeds)
    tab_reliability.tex   violation rate, coverage at plan / at history, by panel
    tab_h1.tex            Spearman correlations of price loss with elasticity error and MAPE
    results_numbers.json  every number quoted in the text
    figs/fig_kt_tradeoff.png, figs/fig_kt_h1.png
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kt_analyze import load, holm  # noqa: E402

# output folder for the manuscript tables/figures (set KT_PAPER_DIR to write elsewhere)
V2 = Path(os.environ.get("KT_PAPER_DIR", str(Path(__file__).resolve().parents[1] / "results" / "paper")))
FAM = ["F1", "F2", "F3", "F4", "F5"]
FAMLAB = {"F1": "F1", "F2": "F2", "F3": "F3", "F4": "F4", "F5": "F5"}
DS = ["dataco", "olist", "synth"]
DSLAB = {"dataco": "DataCo", "olist": "Olist", "synth": "Synth-2026"}
MAIN = ["B1", "B2", "B2m", "B4", "B0f", "A1", "A4", "A4r", "B2c"]
POOLED = ["A1p", "A4p", "A4rp", "A4p_50"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.grid": False,
                     "axes.spines.top": False, "axes.spines.right": False})


def fmt(x, d=1):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def main():
    runs, weeks = load()
    runs = runs[runs.dataset.isin(DS)]
    all_runs = runs
    runs = runs[~runs.model.isin(POOLED)]
    have = [m for m in MAIN if m in set(runs.model)]
    nseeds = runs.groupby(["dataset", "family"]).seed.nunique()
    numbers = {"n_seeds_min": int(nseeds.min()), "n_seeds_max": int(nseeds.max()),
               "cells": int(len(nseeds))}

    # ------------------------------------------------------------ main regret table
    g = runs.groupby(["dataset", "family", "model"]).NR.agg(["mean", "std"]).reset_index()
    lines = []
    for ds in DS:
        first = True
        for fam in FAM:
            sub = g[(g.dataset == ds) & (g.family == fam)]
            if sub.empty:
                continue
            vals = {r.model: r["mean"] for _, r in sub.iterrows()}
            best = min(v for m, v in vals.items() if m in have)
            cells = []
            for m in have:
                v = vals.get(m, np.nan)
                s = fmt(v)
                if not np.isnan(v) and abs(v - best) < 1e-9:
                    s = r"\textbf{" + s + "}"
                cells.append(s)
            lines.append(f"{DSLAB[ds] if first else ''} & {FAMLAB[fam]} & " + " & ".join(cells) + r" \\")
            first = False
        lines.append(r"\midrule")
    avg = g.groupby("model")["mean"].mean()
    lines.append(r"\multicolumn{2}{@{}l}{Mean} & " + " & ".join(fmt(avg.get(m, np.nan)) for m in have) + r" \\")
    numbers["mean_NR"] = {m: float(avg[m]) for m in have if m in avg}
    tab = (r"""\begin{table*}[!t]
\caption{Normalized regret (\%, mean over seeds; lower is better) by panel, true
response family, and model. Bold: lowest in the row. B1 MLP, B2 XGBoost, B2m
monotone XGBoost, B4 BiLSTM, B0f classical structural model, A1 \SIDN, A4
\SIDNCRO ($\rho=1$), A4r \SIDNCRO with rolling recalibration, B2c XGBoost
with the conformal robust layer. Mean: equal weight per panel--family cell.}
\label{tab:main}
\centering\small
\setlength{\tabcolsep}{3.2pt}
\begin{tabular}{@{}ll""" + "r" * len(have) + r"""@{}}
\toprule
Panel & Truth & """ + " & ".join(have) + r""" \\
\midrule
""" + "\n".join(lines[:-1] if lines[-1] == r"\midrule" else lines) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")
    (V2 / "tab_main.tex").write_text(tab, encoding="utf-8")

    # ------------------------------------------------------------ reliability table
    rel_models = [m for m in ["B0f", "A1", "A4", "A4r", "B2c"] if m in set(runs.model)]
    # equal weight per truth family (cell means first), as in the text
    rel = (runs.groupby(["dataset", "family", "model"])[["viol", "cover", "cover_hist"]].mean()
           .groupby(level=["dataset", "model"]).mean().reset_index())
    rl = []
    for ds in DS:
        sub = rel[rel.dataset == ds].set_index("model")
        row = []
        for m in rel_models:
            if m not in sub.index:
                row.append("--")
                continue
            v = 100 * sub.loc[m, "viol"]
            c = sub.loc[m, "cover"]
            ch = sub.loc[m, "cover_hist"]
            row.append(fmt(v, 0) if np.isnan(c) else f"{fmt(v, 0)} / {fmt(100 * c, 0)} / {fmt(100 * ch, 0)}")
        rl.append(f"{DSLAB[ds]} & " + " & ".join(row) + r" \\")
    numbers["reliability"] = rel.to_dict(orient="records")
    tab_rel = (r"""\begin{table}[!t]
\caption{Operational reliability by panel (\%, mean over truth families and
seeds). For nominal models: share of test weeks in which the plan violates a
demand-dependent limit under realized demand. For conformal models: violation
rate / coverage of the band at the planned decision / coverage at the week's
historical decisions (nominal $90\%$).}
\label{tab:reliability}
\centering\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}l""" + "c" * len(rel_models) + r"""@{}}
\toprule
Panel & """ + " & ".join(rel_models) + r""" \\
\midrule
""" + "\n".join(rl) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    (V2 / "tab_reliability.tex").write_text(tab_rel, encoding="utf-8")

    # ------------------------------------------------------------ H1 table and figure
    base = weeks[weeks.model.isin(["B0f", "B1", "B2", "B2m", "B4", "A1"])].replace(
        [np.inf, -np.inf], np.nan).dropna(subset=["elast_err", "mape", "price_loss"])
    h1 = []
    for ds in DS:
        s = base[base.dataset == ds]
        if len(s) < 20:
            continue
        re_ = stats.spearmanr(s.elast_err, s.price_loss).correlation
        rm = stats.spearmanr(s.mape, s.price_loss).correlation
        # partial correlations (rank-based) controlling for the other variable
        r = stats.spearmanr(np.column_stack([s.elast_err, s.mape, s.price_loss])).correlation
        pe = (r[0, 2] - r[0, 1] * r[1, 2]) / np.sqrt((1 - r[0, 1] ** 2) * (1 - r[1, 2] ** 2))
        pm = (r[1, 2] - r[0, 1] * r[0, 2]) / np.sqrt((1 - r[0, 1] ** 2) * (1 - r[0, 2] ** 2))
        h1.append({"dataset": ds, "n": len(s), "rho_elast": re_, "rho_mape": rm, "prho_elast": pe, "prho_mape": pm})
    numbers["h1"] = h1
    hl = [f"{DSLAB[h['dataset']]} & {h['n']} & {h['rho_elast']:.2f} & {h['rho_mape']:.2f} & "
          f"{h['prho_elast']:.2f} & {h['prho_mape']:.2f} \\\\" for h in h1]
    (V2 / "tab_h1.tex").write_text(r"""\begin{table}[!t]
\caption{Spearman correlation of the weekly price loss with the elasticity error
at the plan and with forecast error (MAPE) at historical decisions, across the
nominal models, all truth families, and seeds; partial correlations control for
the other variable.}
\label{tab:h1}
\centering\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}lrcccc@{}}
\toprule
& & \multicolumn{2}{c}{Correlation} & \multicolumn{2}{c}{Partial} \\
\cmidrule(lr){3-4}\cmidrule(l){5-6}
Panel & $n$ & Elast. & MAPE & Elast. & MAPE \\
\midrule
""" + "\n".join(hl) + r"""
\bottomrule
\end{tabular}
\end{table}
""", encoding="utf-8")

    figs = V2 / "figs"
    figs.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.55), sharey=True)
    cols = {"B1": ("#9ecae1", "B1 MLP"), "B2": ("#4292c6", "B2 XGBoost"),
            "B2m": ("#08519c", "B2m monotone XGBoost"), "B4": ("#6a51a3", "B4 BiLSTM"),
            "B0f": ("#525252", "B0f classical structural"), "A1": ("#d94801", "A1 SIDN")}
    for ax, col, lab in [(axes[0], "elast_err", "Elasticity error at the plan"),
                         (axes[1], "mape", "MAPE at historical decisions (%, log scale)")]:
        for m, (c, name) in cols.items():
            t = base[base.model == m]
            if len(t):
                ax.scatter(t[col], t.price_loss, s=3, alpha=0.35, color=c, label=name, linewidths=0)
        ax.set_xlabel(lab)
    axes[1].set_xscale("log")
    axes[0].set_ylabel("Price loss (% of optimal profit)")
    h_, l_ = axes[0].get_legend_handles_labels()
    fig.legend(h_, l_, frameon=False, markerscale=3.5, fontsize=7, loc="upper center",
               bbox_to_anchor=(0.5, 1.0), ncol=6, handletextpad=0.2, columnspacing=0.9)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(figs / "fig_kt_h1.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------ conservatism trade-off
    rho_models = [("A1", 0.0), ("A4_25", 0.25), ("A4_50", 0.5), ("A4_75", 0.75), ("A4", 1.0)]
    fig, (ax_r, ax_v) = plt.subplots(2, 1, figsize=(3.5, 3.7), sharex=True)
    trade = []
    for ds, mk, c in zip(DS, ["o", "s", "^"], ["#1b9e77", "#d95f02", "#7570b3"]):
        pts = []
        for m, rho in rho_models:
            # equal weight per truth family, as in the tables
            sub = runs[(runs.dataset == ds) & (runs.model == m)]
            if len(sub):
                fm = sub.groupby("family")[["viol", "NR"]].mean().mean()
                pts.append((100 * fm["viol"], fm["NR"], rho))
        if pts:
            v, nr_, rho = zip(*pts)
            ax_r.plot(rho, nr_, marker=mk, color=c, label=DSLAB[ds], linewidth=1.2, markersize=4)
            ax_v.plot(rho, v, marker=mk, color=c, label=DSLAB[ds], linewidth=1.2, markersize=4)
            trade.append({"dataset": ds, "points": [{"rho": r_, "viol": x_, "NR": y_} for x_, y_, r_ in pts]})
    numbers["tradeoff"] = trade
    ax_v.set_xlabel(r"Conservatism level $\rho$")
    ax_v.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax_r.set_ylabel("Regret (%)")
    ax_v.set_ylabel("Weeks violated (%)")
    fig.align_ylabels((ax_r, ax_v))
    ax_v.set_ylim(bottom=-1.5)
    ax_v.legend(frameon=False, fontsize=7.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(figs / "fig_kt_tradeoff.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------ coverage at history vs at the plan (H2)
    cm_ = (runs[runs.model.isin(["B2c", "A4", "A4r"])]
           .groupby(["dataset", "family", "model"])[["cover", "cover_hist"]].mean()
           .groupby(level=["dataset", "model"]).mean())
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    cov_models = [("B2c", "XGBoost + CRO (B2c)", "#2171b5"), ("A4", r"SIDN-CRO (A4)", "#d94801"),
                  ("A4r", "SIDN-CRO, rolling (A4r)", "#8c2d04")]
    xt, xl = [], []
    for i, ds in enumerate(DS):
        for j, (m, lab, c) in enumerate(cov_models):
            if (ds, m) not in cm_.index:
                continue
            x = i * 4 + j
            h_, p_ = 100 * cm_.loc[(ds, m), "cover_hist"], 100 * cm_.loc[(ds, m), "cover"]
            ax.annotate("", xy=(x, p_), xytext=(x, h_),
                        arrowprops=dict(arrowstyle="-|>", color=c, lw=1.2, shrinkA=2, shrinkB=2))
            ax.scatter([x], [h_], s=22, facecolors="white", edgecolors=c, linewidths=1.2, zorder=3)
            ax.scatter([x], [p_], s=22, color=c, zorder=3, label=lab if i == 0 else None)
        xt.append(i * 4 + 1)
        xl.append(DSLAB[ds])
    ax.axhline(90, color="0.4", lw=0.8, ls="--")
    ax.text(10.9, 86.5, "nominal 90%", fontsize=6.5, color="0.35", ha="right", va="top")
    ax.set_xticks(xt)
    ax.set_xticklabels(xl)
    ax.set_ylim(-5, 105)
    ax.set_ylabel("Coverage of realized demand (%)")
    ax.legend(frameon=False, fontsize=6.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
              handletextpad=0.3, columnspacing=0.8)
    fig.tight_layout()
    fig.savefig(figs / "fig_kt_coverage.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    numbers["coverage_fig"] = {f"{d}|{m}": {"hist": float(cm_.loc[(d, m), "cover_hist"]),
                                            "plan": float(cm_.loc[(d, m), "cover"])} for d, m in cm_.index}

    # ------------------------------------------------------------ supplementary: pooled variant
    pooled_have = [m for m in POOLED if m in set(all_runs.model)]
    if pooled_have:
        key = ["dataset", "family", "seed"]
        pr = all_runs[all_runs.model.isin(pooled_have)][key].drop_duplicates()
        mt = all_runs.merge(pr, on=key)
        cols_p = [m for m in ["A1", "A1p", "A4", "A4p", "A4rp", "A4p_50"] if m in set(mt.model)]
        gp = mt.groupby(["dataset", "family", "model"]).agg(NR=("NR", "mean"), n=("seed", "nunique"),
                                                          cover=("cover", "mean")).reset_index()
        pl = []
        for ds in DS:
            first = True
            for fam in FAM:
                sub_ = gp[(gp.dataset == ds) & (gp.family == fam)].set_index("model")
                if sub_.empty or "A1p" not in sub_.index:
                    continue
                vals = [fmt(sub_.loc[m, "NR"]) if m in sub_.index else "--" for m in cols_p]
                pl.append(f"{DSLAB[ds] if first else ''} & {fam} & {int(sub_.loc['A1p', 'n'])} & "
                          + " & ".join(vals) + r" \\")
                first = False
            pl.append(r"\midrule")
        cm = gp.groupby("model").NR.mean()
        cv = gp.groupby("model").cover.mean()
        pl.append(r"\multicolumn{3}{@{}l}{Mean regret} & " + " & ".join(fmt(cm.get(m, np.nan)) for m in cols_p) + r" \\")
        # the scaled band of A4p_50 is not a 90% band, so its coverage is not reported
        pl.append(r"\multicolumn{3}{@{}l}{Coverage at plan (\%)} & " + " & ".join(
            fmt(100 * cv[m], 0) if m in cv and not np.isnan(cv[m]) and m != "A4p_50" else "--"
            for m in cols_p) + r" \\")
        (V2 / "supp_tab_pooled.tex").write_text(r"""\begin{table}[!htbp]
\caption{Pooled variant (\SIDN-P) against \SIDN on the runs available for both:
normalized regret (\%) by panel and truth family, with the number of seeds
$n$. A1p \SIDN-P; A4p its conformal robust version ($\rho=1$); A4rp with rolling
recalibration; A4p\_50 with $\rho=0.5$. Coverage: share of test weeks in which
the band contains the realized demand at the planned decision (nominal 90\%;
not reported for A4p\_50, whose band is scaled by $\rho$; see the caveat in
the text on the calibration of \SIDN-P).}
\label{tab:s_pooled}
\centering\small
\begin{tabular}{@{}llr""" + "r" * len(cols_p) + r"""@{}}
\toprule
Panel & Truth & $n$ & """ + " & ".join(c.replace("_", r"\_") for c in cols_p) + r""" \\
\midrule
""" + "\n".join(pl) + r"""
\bottomrule
\end{tabular}
\end{table}
""", encoding="utf-8")
        numbers["pooled_matched"] = {"NR": cm.to_dict(), "cover": cv.to_dict()}

    # ------------------------------------------------------------ paired tests quoted in the text
    tests = []
    for (ds, fam), sub in runs.groupby(["dataset", "family"]):
        piv = sub.pivot(index="seed", columns="model", values="NR")
        for a, b in [("A1p", "B0f"), ("A1p", "A1"), ("A1", "B0f"), ("A1", "B2"), ("A1", "B2m"), ("A1", "B4"),
                     ("A1", "B1"), ("A4", "A1"), ("A4p", "A1p")]:
            if a in piv and b in piv:
                pr = piv[[a, b]].dropna()
                if len(pr) >= 5 and not np.allclose(pr[a], pr[b]):
                    p = stats.wilcoxon(pr[a], pr[b]).pvalue
                    tests.append({"dataset": ds, "family": fam, "a": a, "b": b, "n": len(pr),
                                  "diff": float((pr[a] - pr[b]).mean()), "p": float(p)})
    numbers["tests"] = tests
    numbers["elast"] = runs.groupby("model").elast_err.mean().to_dict()
    numbers["elast_bias"] = runs.groupby("model").elast_bias.mean().to_dict()
    numbers["mape"] = runs.groupby("model").mape.mean().to_dict()
    numbers["price_loss"] = runs.groupby("model").price_loss.mean().to_dict()
    numbers["family_mean_NR"] = runs.groupby(["family", "model"]).NR.mean().unstack().to_dict()
    numbers["dataset_mean_NR"] = runs.groupby(["dataset", "model"]).NR.mean().unstack().to_dict()
    numbers["viol_by_ds_model"] = runs.groupby(["dataset", "model"]).viol.mean().unstack().to_dict()
    (V2 / "results_numbers.json").write_text(json.dumps(numbers, indent=1, default=float), encoding="utf-8")
    print("tables and figures written; seeds per cell:", numbers["n_seeds_min"], "-", numbers["n_seeds_max"],
          "cells:", numbers["cells"])


if __name__ == "__main__":
    main()
