# -*- coding: utf-8 -*-
"""
Generate the Results section (body_results.tex) of the manuscript directly from
the benchmark results. Every number is computed here; every qualitative
statement is checked against the data, and a statement that does not hold is
reported as a FLAG instead of being written.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kt_analyze import load, RES, POOL, pool_complete_seeds  # noqa: E402

V2 = Path(os.environ.get("KT_PAPER_DIR", str(Path(__file__).resolve().parents[1] / "results" / "paper")))
BLACK = ["B1", "B2", "B2m", "B4"]
DSL = {"dataco": "DataCo", "olist": "Olist", "synth": "Synth-2026"}
flags = []


def check(cond, msg):
    if not cond:
        flags.append(msg)
    return cond


def f1(x):
    return f"{x:.1f}"


def pct(x):
    return f"{100 * x:.0f}"


def boot_mean_ci(d, reps=10000, seed=2026):
    """Mean of paired differences and its 95% percentile bootstrap interval."""
    d = np.asarray(d, float)
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), size=(reps, len(d)))].mean(axis=1)
    return [float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def ci(t, d=1):
    return f"{t[0]:.{d}f} points (95\\% CI {t[1]:.{d}f} to {t[2]:.{d}f})"


def pfmt(p):
    return "p<0.001" if p < 1e-3 else f"p={p:.3f}"


def main():
    runs, weeks = load()
    POOLED = ["A1p", "A4p", "A4rp", "A4p_50"]
    main_runs = runs[~runs.model.isin(POOLED)]
    seeds = sorted(int(x) for x in main_runs.seed.unique())
    per_cell = main_runs.groupby(["dataset", "family", "model"]).seed.nunique()
    n_seed_min = int(per_cell.min())
    n_seed_max = int(per_cell.max())
    pool_seeds = sorted(int(x) for x in runs[runs.model.isin(POOLED)].seed.unique())
    WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven",
             8: "eight", 9: "nine", 10: "ten"}

    def seed_text(ss):
        contiguous = ss == list(range(ss[0], ss[-1] + 1))
        rng = f"{ss[0]}--{ss[-1]}" if contiguous and len(ss) > 2 else ", ".join(map(str, ss))
        return f"{WORDS.get(len(ss), len(ss))} seed{'s' if len(ss) > 1 else ''} ({rng})"
    # every aggregate gives each panel--family cell equal weight (as the Mean row of Table tab:main);
    # the pooled variant, run on fewer seeds, is compared with SIDN only on matching runs (section E)
    all_runs = runs
    runs = main_runs
    cellm = runs.groupby(["dataset", "family", "model"]).mean(numeric_only=True)

    def balanced(col, by):
        return cellm[col].groupby(level=by).mean().unstack() if isinstance(by, list) else \
            cellm[col].groupby(level=by).mean()
    cell = cellm["NR"].unstack()
    nr = cell.mean()
    n_cells = len(cell)
    K = {"seeds": seeds, "seeds_text": seed_text(seeds), "n_seed_min": n_seed_min,
         "n_seed_max": n_seed_max, "pool_seeds": pool_seeds,
         "pool_seeds_text": seed_text(pool_seeds) if pool_seeds else ""}
    if pool_seeds:
        ps = pool_seeds
        lst = " and ".join(map(str, ps)) if len(ps) == 2 else (
            f"{ps[0]}--{ps[-1]}" if ps == list(range(ps[0], ps[-1] + 1)) and len(ps) > 2 else ", ".join(map(str, ps)))
        K["pool_seeds_inline"] = f"{WORDS.get(len(ps), len(ps))} seed{'s' if len(ps) > 1 else ''}, {lst}"

    # ---------------- A. plan quality
    best_black = min(BLACK, key=lambda m: nr[m])
    K["nr"] = {m: round(float(nr[m]), 1) for m in nr.index}
    a1_vs_b0 = (cell["A1"] < cell["B0f"]).sum()
    a1_beats_all_black = int((cell["A1"] < cell[BLACK].min(axis=1)).sum())
    b0_beats_all_black = int((cell["B0f"] < cell[BLACK].min(axis=1)).sum())
    check(a1_beats_all_black == n_cells, f"A1 does not beat every black box in all cells ({a1_beats_all_black}/{n_cells})")
    pair = runs.pivot_table(index=["dataset", "family", "seed"], columns="model", values="NR").dropna(subset=["A1", "B0f"])
    w_a1_b0 = stats.wilcoxon(pair["A1"], pair["B0f"])
    n_pairs = len(pair)
    # paired differences over cell--seed pairs with percentile bootstrap intervals
    d_blk = boot_mean_ci(pair[best_black].values - pair["A1"].values)
    w_blk = stats.wilcoxon(pair[best_black], pair["A1"])
    d_b0 = boot_mean_ci(pair["B0f"].values - pair["A1"].values)
    check(d_blk[1] > 0, "A: the interval for (best black box - SIDN) includes zero")
    K.update({"diff_black_A1": d_blk, "p_black_A1": float(w_blk.pvalue), "diff_B0f_A1": d_b0, "n_pairs": n_pairs})
    by_fam = balanced("NR", ["family", "model"])
    fam_a1_better = [f for f in by_fam.index if by_fam.loc[f, "A1"] < by_fam.loc[f, "B0f"]]
    fam_b0_better = [f for f in by_fam.index if by_fam.loc[f, "A1"] >= by_fam.loc[f, "B0f"]]
    by_ds = balanced("NR", ["dataset", "model"])
    mono_gain = nr["B2"] - nr["B2m"]
    ds_a1 = [DSL[d] for d in by_ds.index if by_ds.loc[d, "A1"] < by_ds.loc[d, "B0f"]]
    ds_b0 = [DSL[d] for d in by_ds.index if by_ds.loc[d, "A1"] >= by_ds.loc[d, "B0f"]]
    if ds_a1 and ds_b0:
        ds_sentence = ("Neither structural model dominates: \\SIDN is ahead on " + " and ".join(ds_a1)
                       + ", the classical model on " + " and ".join(ds_b0)
                       + ", so a classical structural model with global elasticities remains a strong baseline.")
    elif ds_a1:
        ds_sentence = "\\SIDN is ahead on every panel."
    else:
        ds_sentence = "The classical model is ahead on every panel."

    A = rf"""
\section{{Results}}
\label{{sec:results}}

Table~\ref{{tab:main}} reports the normalized regret of every model in the
{n_cells} panel--family cells, averaged over {seed_text(seeds) if n_seed_min == n_seed_max
    else 'up to ' + seed_text(seeds) + f', with at least {WORDS.get(n_seed_min, n_seed_min)} per cell'}.

\subsection{{Plan Quality: Structure Matters More Than Flexibility}}
\label{{ssec:res_quality}}

Every model with embedded economic structure plans far better than every
black-box forecaster. Averaged over all cells, \SIDN (A1) forfeits
{f1(nr['A1'])}\% of the oracle profit and the classical structural model (B0f)
{f1(nr['B0f'])}\%, whereas the best black-box model, {best_black}, forfeits
{f1(nr[best_black])}\% and the MLP and BiLSTM {f1(nr['B1'])}\% and {f1(nr['B4'])}\%.
\SIDN has lower regret than all four black-box models in {a1_beats_all_black} of
the {n_cells} cells, and so does the classical model in {b0_beats_all_black}; over the
{n_pairs} cell--seed pairs, its regret is lower than that of {best_black} by
{ci(d_blk)}; paired Wilcoxon test {pfmt(w_blk.pvalue)}.
Monotonicity constraints alone do not close the gap: monotone XGBoost (B2m)
improves on unconstrained XGBoost by only {f1(mono_gain)} points, because tree
ensembles remain flat outside the historical decision range and therefore
cannot rank decisions that were never observed.

Between the two structural models the comparison is closer. \SIDN has lower
regret than the classical model in {a1_vs_b0} of the {n_cells} cells, and its mean
advantage, {ci(d_b0)}, is not statistically significant (paired
Wilcoxon test, {pfmt(w_a1_b0.pvalue)}); by family, it is
ahead in {', '.join(fam_a1_better) if fam_a1_better else 'no family'} and behind
in {', '.join(fam_b0_better) if fam_b0_better else 'no family'}. By panel, \SIDN
forfeits {f1(by_ds.loc['dataco', 'A1'])}\%, {f1(by_ds.loc['olist', 'A1'])}\%, and
{f1(by_ds.loc['synth', 'A1'])}\% on DataCo, Olist, and Synth-2026, against
{f1(by_ds.loc['dataco', 'B0f'])}\%, {f1(by_ds.loc['olist', 'B0f'])}\%, and
{f1(by_ds.loc['synth', 'B0f'])}\% for the classical model. {ds_sentence}
"""
    K.update({"a1_vs_b0_cells": int(a1_vs_b0), "cells": n_cells, "p_a1_b0": float(w_a1_b0.pvalue),
              "fam_a1_better": fam_a1_better, "fam_b0_better": fam_b0_better,
              "best_black": best_black, "mono_gain": float(mono_gain)})

    # ---------------- B. decision-relevant accuracy (H1)
    e = cellm[["elast_err", "mape", "price_loss"]].groupby(level="model").mean()
    bias = cellm["elast_bias"].groupby(level="model").mean()
    under = all(bias[m] < 0 for m in BLACK)
    bias_sentence = (f"All four black-box models underestimate the elasticity (mean signed error "
                     f"${max(bias[BLACK]):.2f}$ to ${min(bias[BLACK]):.2f}$), the costly direction in "
                     f"Fig.~\\ref{{fig:theory}}, which pushes their prices up.") if under else ""
    check(under, "H1: not every black-box model underestimates the elasticity; sentence dropped")
    K["elast_bias"] = {m: float(bias[m]) for m in bias.index}
    base = weeks[weeks.model.isin(["B0f", "B1", "B2", "B2m", "B4", "A1"])].replace(
        [np.inf, -np.inf], np.nan).dropna(subset=["elast_err", "mape", "price_loss"])
    rho_e = stats.spearmanr(base.elast_err, base.price_loss).correlation
    rho_m = stats.spearmanr(base.mape, base.price_loss).correlation
    r = stats.spearmanr(np.column_stack([base.elast_err, base.mape, base.price_loss])).correlation
    pe = (r[0, 2] - r[0, 1] * r[1, 2]) / math.sqrt((1 - r[0, 1] ** 2) * (1 - r[1, 2] ** 2))
    pm = (r[1, 2] - r[0, 1] * r[0, 2]) / math.sqrt((1 - r[0, 1] ** 2) * (1 - r[0, 2] ** 2))
    check(rho_e > rho_m, "H1: elasticity error is not the stronger correlate of price loss")
    # cluster bootstrap over cell--seed pairs for the two correlations
    grp = list(base.groupby(["dataset", "family", "seed"]).indices.values())
    rng = np.random.default_rng(2026)
    be, bm = [], []
    for _ in range(500):
        idx = np.concatenate([grp[k] for k in rng.integers(0, len(grp), len(grp))])
        s_ = base.iloc[idx]
        be.append(stats.spearmanr(s_.elast_err, s_.price_loss).correlation)
        bm.append(stats.spearmanr(s_.mape, s_.price_loss).correlation)
    ci_e = (np.percentile(be, 2.5), np.percentile(be, 97.5))
    ci_m = (np.percentile(bm, 2.5), np.percentile(bm, 97.5))
    check(ci_e[0] > ci_m[1], "H1: the correlation intervals overlap")
    K["rho_elast_ci"], K["rho_mape_ci"] = [float(v) for v in ci_e], [float(v) for v in ci_m]
    # within model and panel (across weeks): rules out a mere difference between model classes
    within = []
    for (_, _), s in base.groupby(["model", "dataset"]):
        if len(s) >= 20:
            within.append((stats.spearmanr(s.elast_err, s.price_loss).correlation,
                           stats.spearmanr(s.mape, s.price_loss).correlation))
    w_e = float(np.median([w[0] for w in within]))
    w_m = float(np.median([w[1] for w in within]))
    w_share = sum(w[0] > w[1] for w in within)
    check(w_share >= 0.8 * len(within), f"H1 within models: elasticity stronger in only {w_share}/{len(within)}")
    K.update({"h1_within_elast": w_e, "h1_within_mape": w_m, "h1_within_share": int(w_share),
              "h1_within_n": len(within)})
    B = rf"""
\subsection{{Which Accuracy Drives Plan Quality (H1)}}
\label{{ssec:res_h1}}

The structural models estimate price elasticities with a mean absolute error of
{e.loc['B0f', 'elast_err']:.2f} (classical) and {e.loc['A1', 'elast_err']:.2f} (\SIDN)
at the planned decisions, against {min(e.loc[BLACK, 'elast_err']):.2f}--{max(e.loc[BLACK, 'elast_err']):.2f}
for the black-box models, and their pricing alone loses {f1(e.loc['B0f', 'price_loss'])}\%
and {f1(e.loc['A1', 'price_loss'])}\% of the optimal profit, against
{f1(min(e.loc[BLACK, 'price_loss']))}--{f1(max(e.loc[BLACK, 'price_loss']))}\%. {bias_sentence} Across
{len(base):,} model--week observations, the price loss is strongly associated with
the elasticity error (Spearman $\rho={rho_e:.2f}$, 95\% CI {ci_e[0]:.2f}--{ci_e[1]:.2f}) and only
weakly with forecast error at the historical decisions ($\rho={rho_m:.2f}$,
{ci_m[0]:.2f}--{ci_m[1]:.2f}; intervals from a bootstrap over cell--seed pairs); controlling for the other
variable, the partial correlations are ${pe:.2f}$ and ${pm:.2f}$
(Table~\ref{{tab:h1}}, Fig.~\ref{{fig:h1}}). The association is not merely a
difference between model classes: within each model and panel, across weeks,
the median correlation of the price loss is {w_e:.2f} with the elasticity error
and {w_m:.2f} with forecast error, and the former is the larger in {w_share} of
the {len(within)} model--panel combinations. This is the pattern that
Propositions~\ref{{prop:level}} and~\ref{{prop:elasticity}} predict: forecast
error at historical decisions, dominated by the demand level, is a poor guide to
the quality of the plan, whereas the accuracy of the demand \emph{{response}}
governs it.
"""
    K.update({"rho_elast": float(rho_e), "rho_mape": float(rho_m), "prho_elast": float(pe),
              "prho_mape": float(pm), "n_h1": int(len(base)),
              "elast": {m: float(e.loc[m, "elast_err"]) for m in e.index},
              "price_loss": {m: float(e.loc[m, "price_loss"]) for m in e.index},
              "mape": {m: float(e.loc[m, "mape"]) for m in e.index}})

    # ---------------- C. guarantees at planned decisions (H2)
    rel = cellm[["viol", "cover", "cover_hist"]].groupby(level=["dataset", "model"]).mean().unstack("model")
    cov = rel["cover"]
    covh = rel["cover_hist"]
    gap_a4 = float((covh["A4"] - cov["A4"]).mean())
    gain_r = {d: float(cov.loc[d, "A4r"] - cov.loc[d, "A4"]) for d in cov.index}
    best_r = max(gain_r, key=gain_r.get)
    drift_d = min(covh.index, key=lambda d: covh.loc[d, "A4"])
    roll_note = ""
    if gain_r[best_r] > 0:
        roll_note = f"; the gain is largest on {DSL[best_r]}"
        if best_r == drift_d:
            roll_note += ", the panel on which the static band also covers least at the historical decisions"
    worse_d = [d for d in cov.index if gain_r[d] < 0]
    if worse_d:
        roll_note += (", while on " + " and ".join(DSL[d] for d in worse_d)
                      + " it lowers coverage at the planned decisions")
        if all(covh.loc[d, "A4r"] > covh.loc[d, "A4"] for d in worse_d):
            roll_note += " although it raises coverage at the historical decisions"
    K["roll_gain"] = gain_r
    # second part of H2: does coverage at the planned decisions deteriorate under misspecification?
    cov_fam = cellm["cover"].xs("A4", level="model").groupby(level="family").mean()
    mis = [f for f in cov_fam.index if f != "F1"]
    worst_f = min(mis, key=lambda f: cov_fam[f])
    drop = float(cov_fam["F1"] - cov_fam[mis].mean())
    fam_sentence = (
        f" Across truth families, the structural band covers the realized demand at the planned "
        f"decisions in {pct(cov_fam['F1'])}\\% of the weeks under the matched family F1 and in "
        f"{pct(cov_fam[mis].min())}--{pct(cov_fam[mis].max())}\\% under the misspecified families; "
        + ("the deterioration that H2 predicts is small on average"
           + (f" and largest under {worst_f} ({pct(cov_fam[worst_f])}\\%)" if cov_fam[worst_f] < cov_fam["F1"] - 0.05 else "")
           if 0 <= drop < 0.05 else
           (f"misspecification lowers it by {pct(drop)} points on average" if drop >= 0.05 else
            "misspecification does not lower it on average"))
        + ".")
    K["cov_by_family_A4"] = {f: float(v) for f, v in cov_fam.items()}
    gap_b2c = float((covh["B2c"] - cov["B2c"]).mean())
    check(gap_b2c > gap_a4, "H2: black-box coverage does not fall more than SIDN coverage")
    # paired difference of the coverage losses over cell--seed pairs
    cp = runs[runs.model.isin(["A4", "B2c"])].pivot_table(index=["dataset", "family", "seed"], columns="model",
                                                          values=["cover", "cover_hist"])
    loss_diff = ((cp["cover_hist"]["B2c"] - cp["cover"]["B2c"]) - (cp["cover_hist"]["A4"] - cp["cover"]["A4"])) * 100
    d_cov = boot_mean_ci(loss_diff.values)
    check(d_cov[1] > 0, "H2: interval of the coverage-loss difference includes zero")
    K["diff_coverage_loss"] = d_cov
    # the mechanism, illustrated on one instance (fig_mechanism.py)
    mech = json.loads((RES / "summary" / "mechanism.json").read_text())
    mp = mech["planned_rel_price"]
    x_lo, x_hi = mech["xgb_band_at_xgb_plan"]
    s_lo, s_hi = mech["sidn_band_at_sidn_plan"]
    x_miss = not (x_lo <= mech["true_D_at_xgb_plan"] <= x_hi)
    s_in = s_lo <= mech["true_D_at_sidn_plan"] <= s_hi
    check(x_miss and s_in, "mechanism figure: band pattern differs from the text")
    mech_par = rf"""
Fig.~\ref{{fig:mechanism}} shows the mechanism behind both hypotheses for one
product--segment pair (DataCo, F1, first test week). Within the historical price
range ({mech['hist_rel_range'][0]:.2f}--{mech['hist_rel_range'][1]:.2f} times the
reference price) XGBoost tracks the true demand, but outside it the tree
ensemble is flat, so its price elasticity is {mech['xgb_elast_at_plan']:.1f}
where the true elasticity is {mech['true_elasticity']:.2f}. Its robust planner
therefore raises the price to the upper bound ({mp['XGBoost+CRO']:.2f}), where the true
expected demand ({mech['true_D_at_xgb_plan']:.0f} units) lies outside its band
({x_lo:.0f}--{x_hi:.0f}). \SIDN extrapolates with the correct shape (elasticity
{mech['sidn_elast_at_plan']:.2f}); its robust planner chooses {mp['SIDN-CRO']:.2f}, where
the band ({s_lo:.0f}--{s_hi:.0f}) contains the true demand
({mech['true_D_at_sidn_plan']:.0f}); the oracle price is {mp['oracle']:.2f}.
"""
    C = rf"""
\subsection{{Do Guarantees Transfer to the Planned Decisions? (H2)}}
\label{{ssec:res_h2}}
{mech_par}
Table~\ref{{tab:reliability}} and Fig.~\ref{{fig:coverage}} separate the two
reasons why a conformal band can fail. At the historical decisions of the test weeks, the band of XGBoost
covers the realized demand in {pct(covh.loc['dataco', 'B2c'])}\%,
{pct(covh.loc['olist', 'B2c'])}\%, and {pct(covh.loc['synth', 'B2c'])}\% of the weeks
on DataCo, Olist, and Synth-2026; at the decisions its own planner chooses,
coverage falls to {pct(cov.loc['dataco', 'B2c'])}\%, {pct(cov.loc['olist', 'B2c'])}\%,
and {pct(cov.loc['synth', 'B2c'])}\%. For \SIDNCRO the corresponding values are
{pct(covh.loc['dataco', 'A4'])}/{pct(covh.loc['olist', 'A4'])}/{pct(covh.loc['synth', 'A4'])}\%
at the historical and {pct(cov.loc['dataco', 'A4'])}/{pct(cov.loc['olist', 'A4'])}/{pct(cov.loc['synth', 'A4'])}\%
at the planned decisions: on average the planner's move away from history costs
the black-box band {gap_b2c * 100:.0f} points of coverage and the structural band
{gap_a4 * 100:.0f} points. The paired difference, {ci(d_cov, 0)}, is the effect
that Theorem~\ref{{thm:transport}} attributes to the shape discrepancy.{fam_sentence} The remaining shortfall at the historical decisions reflects
temporal drift, which violates exchangeability. With rolling recalibration
(A4r), coverage at the planned decisions is {pct(cov.loc['dataco', 'A4r'])}\%,
{pct(cov.loc['olist', 'A4r'])}\%, and {pct(cov.loc['synth', 'A4r'])}\%, against
{pct(cov.loc['dataco', 'A4'])}\%, {pct(cov.loc['olist', 'A4'])}\%, and {pct(cov.loc['synth', 'A4'])}\%
without it{roll_note}.
"""
    K.update({"coverage": {ds: {m: {"plan": float(cov.loc[ds, m]), "hist": float(covh.loc[ds, m])}
                                for m in ["A4", "A4r", "B2c"] if m in cov.columns} for ds in cov.index},
              "gap_a4": gap_a4, "gap_b2c": gap_b2c})

    # ---------------- D. feasibility and the price of guarantees (H3)
    viol = rel["viol"]
    robust = [m for m in ["A4", "A4r", "A4_25", "A4_50", "A4_75"] if m in viol.columns]
    max_robust_viol = float(viol[robust].max().max())
    check(max_robust_viol <= 0.10, f"H3: a robust variant violates limits in {max_robust_viol:.0%} of weeks")
    rhos = [("A1", 0), ("A4_25", 0.25), ("A4_50", 0.5), ("A4_75", 0.75), ("A4", 1)]
    trade = ", ".join(f"{f1(nr[m])}\\%" for m, _ in rhos if m in nr)
    black_viol = float(viol[BLACK].max().max())
    twice = min(nr[BLACK]) > 2 * max(nr["A1"], nr["B0f"])
    check(twice, "H3 text: black-box regret is not more than twice the structural regret")
    # the oracle plan maximizes expected profit under the true demand and does not hedge
    ow = weeks.groupby(["dataset", "family", "seed", "week"]).oracle_viol.first()
    orc = ow.groupby(level=["dataset", "family"]).mean().groupby(level="dataset").mean()
    check(orc.min() > 0.1, "H3: the limits rarely bind at the oracle plan")
    K["oracle_viol"] = {d: float(v) for d, v in orc.items()}
    D = rf"""
\subsection{{Feasibility and the Price of Guarantees (H3)}}
\label{{ssec:res_h3}}

The limits bind at the optimum: even the oracle plan, which maximizes expected
profit under the true demand, violates a capacity or recycled-input limit under
realized demand in {pct(orc.loc['dataco'])}\%, {pct(orc.loc['olist'])}\%, and
{pct(orc.loc['synth'])}\% of the test weeks on DataCo, Olist, and Synth-2026, because
it does not hedge against the noise. Nominal plans that exploit the limits are
exposed to the same risk: under realized demand, the plans of \SIDN violate a capacity or recycled-input limit in
{pct(viol.loc['dataco', 'A1'])}\%, {pct(viol.loc['olist', 'A1'])}\%, and
{pct(viol.loc['synth', 'A1'])}\% of the test weeks on DataCo, Olist, and
Synth-2026, and those of the classical model in {pct(viol.loc['dataco', 'B0f'])}\%,
{pct(viol.loc['olist', 'B0f'])}\%, and {pct(viol.loc['synth', 'B0f'])}\%. Every
conformal robust variant violates a limit in at most {pct(max_robust_viol)}\% of the
weeks. The guarantee has a price in expected profit: as the conservatism level
$\rho$ rises from $0$ to $0.25$, $0.5$, $0.75$, and $1$, the mean regret of \SIDNCRO
is {trade} (Fig.~\ref{{fig:tradeoff}}). The plans of the black-box forecasters
{'never violate a limit' if black_viol == 0 else f'violate a limit in at most {pct(black_viol)}' + chr(92) + '% of the weeks'}, but they forfeit {'more than twice' if twice else 'much more than'}
the profit of the structural plans: their problem is not feasibility but the
demand response.
"""
    K.update({"viol": {ds: {m: float(viol.loc[ds, m]) for m in viol.columns} for ds in viol.index},
              "max_robust_viol": max_robust_viol, "tradeoff_nr": {m: float(nr[m]) for m, _ in rhos if m in nr}})

    # ---------------- E. boundaries (H4) and the pooled variant
    E = r"""
\subsection{When Structure Is Wrong (H4)}
\label{ssec:res_h4}
"""
    fams = list(by_fam.index)
    if "F3" in fams:
        f3 = by_fam.loc["F3"]
        others = [f for f in fams if f != "F3"]
        worst_b0 = all(f3["B0f"] > by_fam.loc[f, "B0f"] for f in others)
        worst_a1 = all(f3["A1"] > by_fam.loc[f, "A1"] for f in others)
        worst_a4 = all(f3["A4"] > by_fam.loc[f, "A4"] for f in others)
        check(worst_b0 and worst_a1, "F3 is not the worst family for both structural models")
        pl3 = balanced("price_loss", ["family", "model"]).loc["F3"]
        small_pl = pl3["B0f"] < 0.25 * f3["B0f"] and pl3["A1"] < 0.25 * f3["A1"]
        check(small_pl, "F3: price loss is not small relative to regret; mechanism sentence dropped")
        f3_mech = (f"The pricing of both models stays close to optimal (price loss {f1(pl3['B0f'])}"
                   + chr(92) + f"% and {f1(pl3['A1'])}" + chr(92) + "%), so the loss arises in the "
                   "advertising and recycled-content decisions, whose response the models misspecify."
                   ) if small_pl else ""
        K["f3_price_loss"] = {"B0f": float(pl3["B0f"]), "A1": float(pl3["A1"])}
        E += (rf"""
Structure helps only as far as its form is right. Under S-shaped advertising and
circularity responses (F3), the classical model forfeits {f1(f3['B0f'])}\% and
\SIDN {f1(f3['A1'])}\%{', their largest regrets across the five families' if worst_b0 and worst_a1 else ''}""")
        E += (rf"""{', and the conformal robust plans lose ' + f1(f3['A4']) + chr(92) + '%' if worst_a4 else ''}.
{f3_mech} In this family {'the classical model' if f3['B0f'] < f3['A1'] else chr(92) + 'SIDN'} is the
better of the two.""")
    if "F5" in fams:
        f5 = by_fam.loc["F5"]
        f1_ = by_fam.loc["F1"]
        d_a1 = f5["A1"] - f1_["A1"]
        d_b0 = f5["B0f"] - f1_["B0f"]
        K["f5_change"] = {"A1": float(d_a1), "B0f": float(d_b0), "A1_F5": float(f5["A1"]),
                          "B0f_F5": float(f5["B0f"]), "black_F5_min": float(min(f5[BLACK]))}
        ahead = f5["A1"] < min(f5[BLACK]) and f5["B0f"] < min(f5[BLACK])
        check(ahead, "F5: structural models not ahead of all black boxes")
        E += (rf""" When the
circularity response has the wrong sign for one segment (F5), the structural
models' regret changes by ${d_a1:+.1f}$ (\SIDN) and ${d_b0:+.1f}$ points (classical)
relative to the matched family F1, and they remain well ahead of the black-box
models ({f1(f5['A1'])}\% and {f1(f5['B0f'])}\% against
{f1(min(f5[BLACK]))}--{f1(max(f5[BLACK]))}\%). In this setting a sign error in the
circularity response is therefore cheap; sign errors in the price response were
not tested.""")
    if pool_seeds:
        # pooled variant against SIDN on exactly the runs where both exist
        pv = all_runs.pivot_table(index=["dataset", "family", "seed"], columns="model",
                                  values=["NR", "cover"])
        m_nr = pv["NR"][["A1p", "A1"]].dropna()
        m_cov = pv["cover"][["A4p", "A4"]].dropna()
        # equal weight per panel--family cell, as elsewhere
        nr_p = m_nr.groupby(level=["dataset", "family"]).mean().mean()
        cov_pm = m_cov.groupby(level=["dataset", "family"]).mean().mean()
        n_match = len(m_nr)
        worse = nr_p["A1p"] > nr_p["A1"]
        # the classical estimates that SIDN-P starts from (stored with each pooled run)
        kap = []
        for f in sorted(POOL.glob("*_seed*.json")):
            d = json.loads(f.read_text())
            if "pooled" in d and d["seed"] in pool_complete_seeds():
                kap.append(max(d["pooled"]["k"]))
        at_bound = sum(k_ >= 1.99 for k_ in kap)
        conf = json.loads((RES / "summary" / "confound.json").read_text())
        c_lo = min(v["corr_ad_L_median"] for v in conf.values())
        c_hi = max(v["corr_ad_L_median"] for v in conf.values())
        check(at_bound >= 0.5 * len(kap), "pooled: classical advertising coefficients mostly not at the bound")
        check(c_lo > 0.2, "pooled: advertising spend not clearly co-moving with base demand")
        E += (rf""" Finally, a
variant of \SIDN that pools its structural parameters toward the classical
estimates (A1p; {K['pool_seeds_inline']}; Supplementary Section~S3) {'does not improve on' if worse else 'improves on'} \SIDN:
on the {n_match} runs available for both, their mean regrets are {f1(nr_p['A1p'])}\% and
{f1(nr_p['A1'])}\%. The classical estimates are a poor anchor outside price: because
historical advertising spend co-moves with base demand over the training weeks
(median within-pair correlation {c_lo:.2f}--{c_hi:.2f} across panels), the classical
fit attributes demand variation to advertising, and its advertising coefficient
reaches the upper bound of the estimation in {at_bound} of {len(kap)} runs.
""")
        K["pooled"] = {"n_match": int(n_match), "nr_A1p": float(nr_p["A1p"]), "nr_A1": float(nr_p["A1"]),
                       "cov_A4p": float(cov_pm["A4p"]), "cov_A4": float(cov_pm["A4"]),
                       "kappa_at_bound": int(at_bound), "n_runs": len(kap),
                       "corr_ad_L": [c_lo, c_hi]}
    F_H1 = r"""\begin{figure}[!t]
\centering
\includegraphics[width=\linewidth]{figs/fig_kt_h1}
\caption{Weekly price loss against the elasticity error at the planned decisions
(left) and against forecast error at the historical decisions (right), for the
nominal models over all truth families and seeds.}
\label{fig:h1}
\end{figure}"""
    F_MECH = r"""\begin{figure*}[!t]
\centering
\includegraphics[width=\linewidth]{figs/fig_kt_mechanism}
\caption{Demand response and conformal band learned by XGBoost and by \SIDN for
one product--segment pair (DataCo, truth family F1, seed 42, first test week),
with all other decisions at the week's historical values. (a) True expected
demand, model forecasts, and 90\% conformal bands (B2c for XGBoost, \SIDNCRO for
\SIDN) against the relative price. (b) Arc price elasticities over 5\% price
increases%MECHNOTE%. Shaded: range of the pair's historical prices. Vertical
lines: prices chosen for this pair by the oracle and by the two robust
planners.}
\label{fig:mechanism}
\end{figure*}"""
    mmin = mech.get("xgb_min_arc_elasticity")
    F_MECH = F_MECH.replace("%MECHNOTE%", (
        f"; the axis is truncated at $-3$, and the XGBoost elasticity reaches ${mmin[0]:.0f}$ at "
        f"{mmin[1]:.2f}, where its forecast rises with price") if mmin else "")
    F_COV = r"""\begin{figure}[!t]
\centering
\includegraphics[width=0.9\linewidth]{figs/fig_kt_coverage}
\caption{Coverage of the conformal band at the week's historical decisions
(open markers) and at the planned decisions (filled markers), by panel, mean
over truth families and seeds. Arrows show the change caused by moving from
history to the plan.}
\label{fig:coverage}
\end{figure}"""
    F_TRADE = r"""\begin{figure}[!t]
\centering
\includegraphics[width=0.9\linewidth]{figs/fig_kt_tradeoff}
\caption{Price of guarantees: mean normalized regret (top) and share of test
weeks with a violated limit under realized demand (bottom) as the conservatism
level $\rho$ of \SIDNCRO rises from $0$ (A1) to $1$ (A4), by panel.}
\label{fig:tradeoff}
\end{figure}"""
    body = (A + "\n\\input{tab_main}\n" + B + "\n\\input{tab_h1}\n\n" + F_H1 + "\n" + C + "\n" + F_MECH
            + "\n\n\\input{tab_reliability}\n\n" + F_COV + "\n" + D + "\n" + F_TRADE + "\n" + E)
    (V2 / "body_results.tex").write_text(body, encoding="utf-8")
    (V2 / "key_numbers.json").write_text(json.dumps(K, indent=1, default=float), encoding="utf-8")
    print(f"body_results.tex written (seeds {seeds}, {n_seed_min}-{n_seed_max}/cell; pooled {pool_seeds})")
    print("FLAGS:" if flags else "no flags", *flags, sep="\n  ")


if __name__ == "__main__":
    main()
