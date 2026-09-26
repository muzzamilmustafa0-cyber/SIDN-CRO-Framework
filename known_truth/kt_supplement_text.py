# -*- coding: utf-8 -*-
"""
Generate the result-dependent sections of the Supplementary Material
(supp_generated.tex): S3 the pooled variant SIDN-P, S4 additional results.
Every number is computed from the result files.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kt_analyze import load, RES, POOL, pool_complete_seeds  # noqa: E402

V2 = Path(os.environ.get("KT_PAPER_DIR", str(Path(__file__).resolve().parents[1] / "results" / "paper")))
DS = ["dataco", "olist", "synth"]
DSL = {"dataco": "DataCo", "olist": "Olist", "synth": "Synth-2026"}
FAM = ["F1", "F2", "F3", "F4", "F5"]
BLACK = ["B1", "B2", "B2m", "B4"]
POOLED = ["A1p", "A4p", "A4rp", "A4p_50"]


def f1(x):
    return "--" if x is None or np.isnan(x) else f"{x:.1f}"


def f2(x):
    return "--" if x is None or np.isnan(x) else f"{x:.2f}"


def main():
    runs, _ = load()
    main_runs = runs[~runs.model.isin(POOLED)]
    K = json.loads((V2 / "key_numbers.json").read_text(encoding="utf-8"))

    # ------------------------------------------------------------------ S3 pooled variant
    est = []
    for f in sorted(POOL.glob("*_seed*.json")):
        d = json.loads(f.read_text())
        if "pooled" in d and d["seed"] in pool_complete_seeds():
            est.append({"ds": d["dataset"], "lam": d["lam"], "beta": d["pooled"]["beta"],
                        "k": d["pooled"]["k"], "alpha": d["pooled"]["alpha"]})
    n_est = len(est)
    betas = [e["beta"] for e in est]
    k_bound = sum(max(e["k"]) >= 1.99 for e in est)
    k_all = sum(min(e["k"]) >= 1.99 for e in est)
    a_lo = sum(e["alpha"] <= 0.01 for e in est)
    a_hi = sum(e["alpha"] >= 4.99 for e in est)
    lam_counts = {lam: sum(e["lam"] == lam for e in est) for lam in (1e-3, 1e-2, 1e-1)}
    conf = json.loads((RES / "summary" / "confound.json").read_text())
    conf_rows = "\n".join(
        f"{DSL[d]} & {conf[d]['corr_ad_L_median']:.2f} [{conf[d]['corr_ad_L_min']:.2f}, "
        f"{conf[d]['corr_ad_L_max']:.2f}] & {conf[d]['corr_u_L_median']:.2f} & "
        f"{conf[d]['u_min']:.3f}--{conf[d]['u_max']:.3f} \\\\" for d in DS)
    pooled = K.get("pooled", {})
    # cells in which SIDN-P is ahead of SIDN on the matched runs
    key = ["dataset", "family", "seed"]
    pr = runs[runs.model == "A1p"][key].drop_duplicates()
    mt = runs.merge(pr, on=key)
    cmp_ = mt[mt.model.isin(["A1", "A1p"])].groupby(["dataset", "family", "model"]).NR.mean().unstack()
    ahead = [f"{DSL[d]} {f}" for (d, f) in cmp_.index if cmp_.loc[(d, f), "A1p"] < cmp_.loc[(d, f), "A1"]]
    ahead_text = (f"; it is ahead in {len(ahead)} of {len(cmp_)} cells ({', '.join(ahead)})" if ahead
                  else "; it is behind in every cell")
    supp_tab =(V2 / "supp_tab_pooled.tex").read_text(encoding="utf-8") if (V2 / "supp_tab_pooled.tex").exists() else ""

    S3 = rf"""
\section{{Pooled Variant (\SIDN-P)}}
\label{{supp:pooled}}

\textbf{{Motivation and method.}} Proposition~3 of the main text implies that the
profit lost to an elasticity error grows with its square, so noisy
context-specific parameters can be costly when historical decisions vary
little. \SIDN-P therefore (i) initializes the structural output heads
($\beta$, $k_s,k_m,k_r$, $\alpha$) at the global estimates of the classical
structural model B0f fitted on the training weeks; (ii) penalizes the
context-dependent deviations from these values with an $L_2$ penalty
$\lambda\sum\|W\|_2^2$ on the weights of the structural heads; and (iii) selects
$\lambda\in\{{10^{{-3}},10^{{-2}},10^{{-1}}\}}$ and the stopping epoch by the loss
on the validation weeks. Architecture, loss, and the monotonicity guarantee
are those of \SIDN. The conformal robust versions (A4p, A4rp, A4p\_50) use the
same conformal layer as \SIDNCRO.

\textbf{{Caveat on the band.}} The validation weeks are also the calibration
weeks of the conformal layer. Because \SIDN-P selects $\lambda$ and its stopping
epoch on these weeks, its calibration scores are optimistically small and its
band can undercover for reasons unrelated to the shape discrepancy. The
coverage of A4p and A4rp is therefore reported for completeness only and is not
used as evidence in the main text. \SIDN (A1) stops on the training loss and
does not have this problem.

\textbf{{The classical anchor.}} Over the {n_est} runs, the classical fit
estimates the price elasticity well ($\hat\beta\in[{min(betas):.2f},{max(betas):.2f}]$
against true pair values in $[1.6,2.8]$), but not the other responses: at least
one advertising coefficient reaches the upper bound of the estimation
($\hat\kappa=2$, against true values of $0.02$--$0.08$) in {k_bound} runs, and all
three do so in {k_all}; the circularity coefficient ends at the lower bound $0$
in {a_lo} runs and at the upper bound $5$ in {a_hi} (true values $0.3$--$1.0$). The
cause is visible in the data (Table~\ref{{tab:s_confound}}): historical advertising
spend was generated from advertising-to-sales ratios (Section~\ref{{supp:data}}), a
common budgeting practice, so it co-moves with base demand, and a model
without time effects attributes the co-movement to advertising; the
recycled-content shares vary within a narrow range, which leaves the
circularity coefficient weakly identified. \SIDN's context encoder receives
calendar and market-context inputs that can absorb part of this co-movement.
The penalty selected on the validation weeks was $\lambda=10^{{-3}}$ in
{lam_counts[1e-3]} runs, $10^{{-2}}$ in {lam_counts[1e-2]}, and $10^{{-1}}$ in
{lam_counts[1e-1]}.

\begin{{table}}[!htbp]
\caption{{Historical decisions and base demand over the training weeks:
within-pair correlation between $\log(1+\text{{advertising spend}})$ and the
logarithm of base demand (median and range over the 15 pairs), the median
correlation of the mean recycled-content share with the logarithm of base
demand, and the range of the recycled-content share.}}
\label{{tab:s_confound}}
\centering\small
\begin{{tabular}}{{@{{}}lccc@{{}}}}
\toprule
Panel & Advertising & Recycled share & Share range \\
\midrule
{conf_rows}
\bottomrule
\end{{tabular}}
\end{{table}}

\textbf{{Results.}} Table~\ref{{tab:s_pooled}} compares \SIDN-P with \SIDN on the runs
available for both{f" ({pooled['n_match']} runs)" if pooled else ""}. Averaged with equal
weight per panel--family cell, \SIDN-P forfeits {f1(pooled.get('nr_A1p', np.nan))}\% of the
oracle profit against {f1(pooled.get('nr_A1', np.nan))}\% for \SIDN{ahead_text}.

{supp_tab}
"""

    # ------------------------------------------------------------------ S4 additional results
    sel = ["B2m", "B0f", "A1", "A4", "A4r"]
    g = main_runs.groupby(["dataset", "family", "model"]).NR.agg(["mean", "std", "count"])
    rows = []
    for ds in DS:
        first = True
        for fam in FAM:
            if (ds, fam, "A1") not in g.index:
                continue
            cells = []
            for m in sel:
                if (ds, fam, m) in g.index:
                    r = g.loc[(ds, fam, m)]
                    cells.append(f"{r['mean']:.1f} ({r['std']:.1f})" if r["count"] > 1 else f"{r['mean']:.1f}")
                else:
                    cells.append("--")
            n = int(g.loc[(ds, fam, "A1"), "count"])
            rows.append(f"{DSL[ds] if first else ''} & {fam} & {n} & " + " & ".join(cells) + r" \\")
            first = False
        rows.append(r"\midrule")
    rows = rows[:-1]
    tab_sd = (r"""\begin{table}[!htbp]
\caption{Normalized regret (\%): mean (standard deviation) over seeds, with the
number of seeds $n$, for the best black-box model (B2m), the two structural
models, and \SIDNCRO without (A4) and with (A4r) rolling recalibration.}
\label{tab:s_sd}
\centering\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}llr""" + "c" * len(sel) + r"""@{}}
\toprule
Panel & Truth & $n$ & """ + " & ".join(sel) + r""" \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

    # accuracy metrics by panel and model (cell-balanced)
    acc_models = ["B1", "B2", "B2m", "B4", "B0f", "A1"]
    cm = main_runs.groupby(["dataset", "family", "model"])[["elast_err", "elast_bias", "price_loss", "mape"]].mean()
    by = cm.groupby(level=["dataset", "model"]).mean()
    arows = []
    for ds in DS:
        first = True
        for m in acc_models:
            if (ds, m) not in by.index:
                continue
            r = by.loc[(ds, m)]
            arows.append(f"{DSL[ds] if first else ''} & {m} & {f2(r['elast_err'])} & ${f2(r['elast_bias'])}$ & "
                         f"{f1(r['price_loss'])} & {f1(r['mape'])} \\\\")
            first = False
        arows.append(r"\midrule")
    arows = arows[:-1]
    tab_acc = (r"""\begin{table}[!htbp]
\caption{Accuracy by panel and nominal model (mean over truth families and
seeds): mean absolute and mean signed error of the price elasticity at the
planned decision (signed error: model minus truth, on the absolute
elasticity), price loss (\% of the optimal profit), and forecast error at the
historical decisions (MAPE, \%).}
\label{tab:s_acc}
\centering\small
\begin{tabular}{@{}llcccc@{}}
\toprule
Panel & Model & $|$Elast.\ error$|$ & Elast.\ bias & Price loss & MAPE \\
\midrule
""" + "\n".join(arows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

    # run-level win counts
    piv = main_runs.pivot_table(index=["dataset", "family", "seed"], columns="model", values="NR")
    n_runs = len(piv.dropna(subset=["A1", "B0f"] + BLACK))
    pv = piv.dropna(subset=["A1", "B0f"] + BLACK)
    wins = {
        "A1 < every black box": int((pv["A1"] < pv[BLACK].min(axis=1)).sum()),
        "B0f < every black box": int((pv["B0f"] < pv[BLACK].min(axis=1)).sum()),
        "A1 < B0f": int((pv["A1"] < pv["B0f"]).sum()),
    }
    if "A4" in pv:
        wins["A4 < every black box"] = int((pv["A4"] < pv[BLACK].min(axis=1)).sum())
    win_rows = "\n".join(f"{k} & {v} of {n_runs} \\\\" for k, v in wins.items())
    tab_win = (r"""\begin{table}[!htbp]
\caption{Run-level comparisons: number of runs (panel, truth family, seed) in
which the first model has lower normalized regret.}
\label{tab:s_wins}
\centering\small
\begin{tabular}{@{}lc@{}}
\toprule
Comparison & Runs \\
\midrule
""" + win_rows + r"""
\bottomrule
\end{tabular}
\end{table}
""")

    S4 = rf"""
\section{{Additional Results}}
\label{{supp:results}}

Table~\ref{{tab:s_sd}} gives the seed-to-seed variation of the normalized regret
for the main models, Table~\ref{{tab:s_acc}} the accuracy metrics by panel and
model, and Table~\ref{{tab:s_wins}} run-level comparisons. With {K['seeds_text']}
per cell, per-cell rank tests cannot reach conventional significance levels
(the smallest attainable two-sided $p$-value of a Wilcoxon signed-rank test with
{K['n_seed_max']} pairs is ${2 / 2 ** K['n_seed_max']:.4g}$); the main text therefore tests over all cell--seed
pairs, and the run-level counts show how consistent the differences are.

{tab_sd}
{tab_acc}
{tab_win}
"""
    (V2 / "supp_generated.tex").write_text(S3 + S4, encoding="utf-8")
    K2 = {"n_est": n_est, "k_bound": k_bound, "k_all": k_all, "a_lo": a_lo, "a_hi": a_hi,
          "lam_counts": {str(k): v for k, v in lam_counts.items()}, "wins": wins, "n_runs": n_runs}
    (V2 / "supp_numbers.json").write_text(json.dumps(K2, indent=1), encoding="utf-8")
    print("supp_generated.tex written:", K2)


if __name__ == "__main__":
    main()
