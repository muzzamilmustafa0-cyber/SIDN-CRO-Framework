# -*- coding: utf-8 -*-
"""
Generate the result-dependent prose outside the Results section: the summary
paragraph of the Introduction (results_intro.tex), the Discussion, Limitations,
and Conclusion (body_discussion.tex), the abstract (abstract.tex), and the
managerial relevance statement (mrs.tex). Every number comes from
key_numbers.json / results_numbers.json written by kt_results_text.py and
kt_tables.py; every qualitative claim is checked and reported as a FLAG if it
does not hold.
"""
from __future__ import annotations

import json
import os
import re

import numpy as np
from pathlib import Path

V2 = Path(os.environ.get("KT_PAPER_DIR", str(Path(__file__).resolve().parents[1] / "results" / "paper")))
(V2 / "figs").mkdir(parents=True, exist_ok=True)   # output folder of the tables, figures, and text
BLACK = ["B1", "B2", "B2m", "B4"]
DSL = {"dataco": "DataCo", "olist": "Olist", "synth": "Synth-2026"}
flags = []
# journal-specific wording only (numbers and checks are identical): "ejor" (default; European Journal
# of Operational Research: no unexplained acronyms in the abstract) or "tem" (an earlier version)
VENUE = os.environ.get("KT_VENUE", "ejor").lower()
EJOR = VENUE == "ejor"


def check(cond, msg):
    if not cond:
        flags.append(msg)
    return cond


def f1(x):
    return f"{x:.1f}"


def f0(x):
    return f"{x:.0f}"


def words(t):
    t = re.sub(r"\\[a-zA-Z]+", " ", t)
    return len(re.findall(r"[A-Za-z0-9][A-Za-z0-9'\-]*", t))


def main():
    K = json.loads((V2 / "key_numbers.json").read_text(encoding="utf-8"))
    R = json.loads((V2 / "results_numbers.json").read_text(encoding="utf-8"))
    nr = R["mean_NR"]                       # cell-balanced, identical to the Mean row of Table tab:main
    for m in ("A1", "B0f"):
        check(abs(nr[m] - K["nr"][m]) < 0.051, f"table/text mismatch for {m}: {nr[m]} vs {K['nr'][m]}")
    struct_lo, struct_hi = min(nr["A1"], nr["B0f"]), max(nr["A1"], nr["B0f"])
    black_lo, black_hi = min(nr[m] for m in BLACK), max(nr[m] for m in BLACK)
    check(black_lo > struct_hi, "a black-box model is not worse than both structural models")
    h1 = {h["dataset"]: h for h in R["h1"]}
    re_lo, re_hi = min(h["rho_elast"] for h in h1.values()), max(h["rho_elast"] for h in h1.values())
    rm_lo, rm_hi = min(h["rho_mape"] for h in h1.values()), max(h["rho_mape"] for h in h1.values())
    h1_all = check(all(h["rho_elast"] > h["rho_mape"] for h in h1.values()),
                   "H1: elasticity correlation not above the MAPE correlation in every panel")
    cov = K["coverage"]
    b2c_h = [cov[d]["B2c"]["hist"] for d in DSL]
    b2c_p = [cov[d]["B2c"]["plan"] for d in DSL]
    a4_h = [cov[d]["A4"]["hist"] for d in DSL]
    a4_p = [cov[d]["A4"]["plan"] for d in DSL]
    gap_a4, gap_b2c = 100 * K["gap_a4"], 100 * K["gap_b2c"]
    check(gap_b2c > 2 * gap_a4, "H2: black-box coverage loss not much larger than the structural one")
    # panel on which the static structural band keeps nominal coverage at the planned decisions
    nominal_ds = [d for d in DSL if cov[d]["A4"]["plan"] >= 0.895 and cov[d]["A4"]["hist"] >= 0.895]
    trade = {t["dataset"]: {p["rho"]: p for p in t["points"]} for t in R["tradeoff"]}
    viol = K["viol"]
    max_nom_viol = max(viol[d]["A1"] for d in DSL)
    robust_max = K["max_robust_viol"]
    check(robust_max < max_nom_viol, "H3: robust violations not below nominal violations")
    p_a1_b0 = K["p_a1_b0"]
    check(p_a1_b0 > 0.05, "text says the two structural models are not significantly different")
    pooled = K.get("pooled")
    el = K["elast"]
    f5_ok = "F5" in K.get("fam_b0_better", []) or "F5" in K.get("fam_a1_better", [])

    # ------------------------------------------------------------------ trade-off sentences
    def tr(ds, rho):
        return trade[ds].get(rho) or trade[ds].get(float(rho))
    tsent = []
    for ds in ("synth", "olist"):
        p0, p25 = tr(ds, 0.0), tr(ds, 0.25)
        if p0 and p25 and p0["viol"] > 1 and p25["viol"] < p0["viol"]:
            if not tsent:
                tsent.append(f"on {DSL[ds]}, $\\rho=0.25$ reduced the share of weeks with a violated limit "
                             f"from {f0(p0['viol'])}\\% to {f0(p25['viol'])}\\% at a cost of "
                             f"{f1(p25['NR'] - p0['NR'])} points of regret")
            else:
                tsent.append(f"on {DSL[ds]}, it reduced that share from {f0(p0['viol'])}\\% to "
                             f"{f0(p25['viol'])}\\% at a cost of {f1(p25['NR'] - p0['NR'])} points")
    d0, d1 = tr("dataco", 0.0), tr("dataco", 1.0)
    dataco_free = d0 and d0["viol"] < 1
    if dataco_free:
        tsent.append(f"on DataCo, where the nominal plans violated no limit, the robust layer only "
                     f"cost profit (regret {f1(d0['NR'])}\\% at $\\rho=0$ and {f1(d1['NR'])}\\% at $\\rho=1$)")
    trade_text = "; ".join(tsent)

    # ------------------------------------------------------------------ Introduction summary
    intro = rf"""
The experiments support the theory where it makes firm predictions and
delimit it where it does not. Across the {K['cells']} panel--family
combinations, the nominal structural models forfeit about
{f0(struct_lo)}--{f0(struct_hi)}\% of the oracle profit, against
{f0(black_lo)}--{f0(black_hi)}\% for four black-box forecasters; the weekly price
loss tracks the elasticity error rather than forecast error; and moving from
historical to planned decisions costs the conformal band of a black-box
forecaster {f0(gap_b2c)} points of coverage on average, against {f0(gap_a4)}
for \SIDNCRO. A neural network with context-dependent elasticities, however,
does not significantly outperform a classical structural model with global
elasticities, and a sign error in the circularity response proves cheap. The
remainder of the paper reviews related work (Section~\ref{{sec:lit}}),
presents the planning problem and \SIDNCRO (Section~\ref{{sec:method}}),
develops the theory (Section~\ref{{sec:theory}}), describes the evaluation
design (Section~\ref{{sec:exp}}), reports the results
(Section~\ref{{sec:results}}), and discusses their implications and limitations
(Sections~\ref{{sec:discussion}} and~\ref{{sec:limits}}).
"""

    # ------------------------------------------------------------------ Discussion
    h2_nominal = (f" and keeps its nominal level on {DSL[nominal_ds[0]]}, where it is also nominal at the "
                  "historical decisions") if len(nominal_ds) == 1 else ""
    # sensitivity analysis (kt_sens_text.py), if available
    sens_flex, sens_mgr, sens_lim = "", "", ""
    if (V2 / "sens_numbers.json").exists():
        _sn = json.loads((V2 / "sens_numbers.json").read_text(encoding="utf-8"))
        _k = len(_sn.get("seeds_used", [])) or 3
        _w = {1: "one seed", 2: "two seeds", 3: "three seeds", 4: "four seeds", 5: "five seeds"}.get(_k, f"{_k} seeds")
        sens_lim = (f" The sensitivity analysis uses {_w} and a reduced set of models, and it"
                    " does not vary the tightness of the operational limits or the length of the"
                    " calibration window.")
    sf = V2 / "sens_numbers.json"
    if sf.exists():
        SN = json.loads(sf.read_text(encoding="utf-8"))
        a_lo, a_hi = SN["sp0.05_A1"], SN["sp0.3_A1"]
        if SN["elast_decreasing"]["A1"] and SN["elast_decreasing"]["B0f"]:
            dd = [SN["a1_minus_b0f"][k] for k in ("0.05", "0.15", "0.3")]
            grows = dd[0] > dd[1] > dd[2]
            sens_flex = (f" The sensitivity analysis supports this reading: as the historical price variation "
                         f"rises from $\\sigma_p=0.05$ to $0.30$, the elasticity error of \\SIDN falls from "
                         f"{a_lo['elast_err']:.2f} to {a_hi['elast_err']:.2f}"
                         + (f", and its regret relative to the classical model moves from ${dd[0]:+.1f}$ to "
                            f"${dd[2]:+.1f}$ points" if grows else "")
                         + " (Section~\\ref{ssec:res_sens}).")
        else:
            flags.append("sensitivity: elasticity error not decreasing for both structural models; sentence dropped")
        if SN["regret_decreasing"]["A1"]:
            sens_mgr = rf"""

\emph{{Generate the variation that the model needs.}} Elasticities are
identified only from the variation in past decisions. In the benchmark, raising
the historical price variation from $\sigma_p=0.05$ to $0.30$ lowered the regret
of \SIDN from {f1(a_lo['NR'])}\% to {f1(a_hi['NR'])}\% (Section~\ref{{ssec:res_sens}}). Rotating
promotions and planned price tests supply this variation at a controlled cost
and should be treated as an investment in the planning model, not only as a
marketing decision."""
    cf = K["cov_by_family_A4"]
    mis_drop = cf["F1"] - np.mean([v for f, v in cf.items() if f != "F1"]) if "F1" in cf else 0.0
    h2_mis = (f"misspecification lowers it by {f0(100 * mis_drop)} points on average" if mis_drop >= 0.005 else
              "misspecification does not lower it on average")
    pooled_sent = ""
    if pooled:
        pooled_sent = (f"; moreover, shrinking \\SIDN toward the classical estimates, which are poorly "
                       f"identified outside price, made it worse "
                       f"({f1(pooled['nr_A1p'])}\\% against {f1(pooled['nr_A1'])}\\% on the same runs)")
    disc = rf"""
\section{{Discussion}}
\label{{sec:discussion}}

\subsection{{Theoretical Implications}}
\label{{ssec:theory_impl}}

Table~\ref{{tab:hyp}} summarizes the tests. The experiments support H1 and H3
and do not support H4. They support H2 only in part: the structural band loses far less coverage than the black-box band
when the plan moves away from history, as Theorem~\ref{{thm:transport}}
predicts, but it reaches the nominal level only where calibration and planning
weeks do not drift apart, and {h2_mis}. Three implications follow.

\input{{tab_hyp}}

\emph{{Which accuracy matters.}} Propositions~\ref{{prop:level}}
and~\ref{{prop:elasticity}} predict that the level of a multiplicative forecast
is irrelevant for pricing and that profit is lost through the elasticity error.
The benchmark confirms this at the level of individual planning weeks: within
{'every' if h1_all else 'each'} panel, the price loss correlates at
{re_lo:.2f}--{re_hi:.2f} with the elasticity error and at {rm_lo:.2f}--{rm_hi:.2f}
with forecast error at historical decisions (Table~\ref{{tab:h1}}). Hold-out
forecast accuracy, the standard criterion for comparing forecasters in practice
and in forecasting competitions~\citep{{Makridakis2018}}, therefore certifies a
property that matters little for the quality of the prices. The result carries the argument of
decision-focused learning~\citep{{Elmachtoub2022}} from the loss used for
training to the criterion used for model selection, and it is consistent with
the finding of~\citet{{BesbesZeevi2015}} that a misspecified demand model can
price well when it captures the local demand response.

\emph{{Which guarantees survive planning.}} Conformal guarantees are usually
presented as model-agnostic~\citep{{Vovk2005,Angelopoulos2022}}, and conformal
robust optimization inherits this property~\citep{{JohnstoneCox2021,Patel2024aistats}}.
Theorem~\ref{{thm:transport}} shows that the guarantee is model-agnostic only at
the decisions that generated the calibration data; at a planned decision it
depends on the shape discrepancy, and hence on the model. The experiments make
the distinction visible: at the historical decisions the black-box band covers
the realized demand in {f0(100 * min(b2c_h))}--{f0(100 * max(b2c_h))}\% of the weeks and
the structural band in {f0(100 * min(a4_h))}--{f0(100 * max(a4_h))}\%, but at the planned decisions the black-box band loses {f0(gap_b2c)} points of
coverage on average, whereas the structural band loses {f0(gap_a4)}{h2_nominal}.
For decisions chosen by an optimizer, embedded structure is therefore not an
alternative to distribution-free uncertainty quantification but its
precondition. The shortfall that remains at the historical decisions is a
different failure, temporal drift, which Theorem~\ref{{thm:transport}} excludes
through its exchangeability assumption; weighted and adaptive conformal
methods~\citep{{Tibshirani2019,Gibbs2021}} address that failure, not the
decision shift.

\emph{{Flexibility versus structure.}} Context-dependent elasticities did not
buy a significant improvement over a classical structural model with global
elasticities (lower regret in {K['a1_vs_b0_cells']} of {K['cells']} cells,
$p={p_a1_b0:.2f}$), and the more flexible model did not estimate elasticities more
accurately (mean absolute error {el['A1']:.2f} against {el['B0f']:.2f}){pooled_sent}.
This pattern is consistent with the view that the returns to flexibility in
decision-dependent planning are bounded by the decision variation that the
history contains: Proposition~\ref{{prop:elasticity}} ties the profit loss to
the elasticity error, and flexibility without identifying variation does not
reduce it.{sens_flex} The value of {'a machine-learning' if EJOR else 'an AI-based'} demand model for planning lies first in how
it encodes the demand response and only then in its capacity to fit the data.

\subsection{{Managerial Implications}}
\label{{ssec:mgr_impl}}

{'Research on supply chain analytics' if EJOR else 'Work in this journal'} has examined how analytics and AI applications shape the
responsiveness, resilience, and risk management of supply
chains~\citep{{Stahl2023tem,Virmani2024tem,UlHaq2026tem,Jia2026tem}}. The present
results bear on a question that precedes these benefits: how the analytics that
choose plans should be selected, audited, and operated. {'Five' if sens_mgr else 'Four'} implications
follow for organizations that let {'machine-learning' if EJOR else 'AI-based'} pipelines plan prices, promotions,
and recycled content.

\emph{{Audit the response, not the fit.}} When a forecaster feeds an
optimizer, its hold-out error is a weak indicator of plan quality. Managers
should request the elasticities that the model implies at the proposed plan,
compare them with the evidence from price tests or with the elasticities of a
simple structural model, and treat large disagreements as a warning. Post-hoc
explanation methods can make black-box models more transparent~\citep{{Zhan2024tem}};
for planning, the explanation that matters is the demand response at the
proposed plan, which a structural model exposes directly.

\emph{{Do not carry black-box uncertainty bands over to new decisions.}} A
band that covered {f0(100 * min(b2c_h))}--{f0(100 * max(b2c_h))}\% of the weeks at the
historical decisions covered {f0(100 * min(b2c_p))}--{f0(100 * max(b2c_p))}\% at the
plans chosen with it. Commitments based on such bands, such as capacity
reservations or recycled-input contracts, carry risks that the band does not
show.

\emph{{Set the conservatism level according to how tightly limits bind.}} In
the benchmark, {trade_text}. The level $\rho$ should be chosen by comparing
the expected penalty of a violated commitment with this price, and the band
should be recalibrated when conditions drift, with coverage monitored at the
implemented plans rather than at historical decisions.

\emph{{Keep a simple structural benchmark.}} A classical model with global
elasticities matched the neural structural model on average. A more flexible
model should be adopted only if it improves on this benchmark in the quality
of validation plans, not merely in forecast accuracy.{sens_mgr}
"""
    # ------------------------------------------------------------------ hypothesis summary table
    f5c = K["f5_change"]
    ci_e = K["rho_elast_ci"]
    dcov = K["diff_coverage_loss"]
    v1 = "Supported" if (K["rho_elast"] > K["rho_mape"] and ci_e[0] > K["rho_mape_ci"][1]) else "Not supported"
    v3 = "Supported" if robust_max < max_nom_viol else "Not supported"
    v4 = ("Not supported" if (f5c["A1_F5"] < f5c["black_F5_min"] and f5c["B0f_F5"] < f5c["black_F5_min"]
                              and max(f5c["A1"], f5c["B0f"]) < 5) else "Supported")
    nominal_txt = (f"nominal only on {' and '.join(DSL[d] for d in nominal_ds)}" if nominal_ds
                   else "below nominal on every panel")
    tab_hyp = rf"""\begin{{table}}[!t]
\caption{{Summary of the hypothesis tests (Section~\ref{{sec:results}}).}}
\label{{tab:hyp}}
\centering\footnotesize
\renewcommand{{\arraystretch}}{{1.15}}
\setlength{{\tabcolsep}}{{3pt}}
\begin{{tabular}}{{@{{}}p{{0.27\linewidth}}p{{0.5\linewidth}}p{{0.15\linewidth}}@{{}}}}
\toprule
Hypothesis & Evidence & Verdict \\
\midrule
H1: price regret tracks elasticity error, not forecast error &
Spearman {K['rho_elast']:.2f} (95\% CI {ci_e[0]:.2f}--{ci_e[1]:.2f}) against {K['rho_mape']:.2f}; larger in {K['h1_within_share']} of {K['h1_within_n']} model--panel combinations & {v1} \\
H2: coverage at planned decisions stays near nominal for the correct shape and deteriorates with misspecification &
Coverage loss {f0(gap_a4)} against {f0(gap_b2c)} points for a black-box band (difference {f0(dcov[0])}, 95\% CI {f0(dcov[1])}--{f0(dcov[2])}); {nominal_txt}; {f0(100 * mis_drop)} points lower under misspecification & Partly supported \\
H3: conformal robust plans reduce realized violations &
At most {f0(100 * robust_max)}\% of weeks against up to {f0(100 * max_nom_viol)}\% for nominal plans; regret {f1(K['tradeoff_nr']['A1'])}\% at $\rho=0$ and {f1(K['tradeoff_nr']['A4'])}\% at $\rho=1$ & {v3} \\
H4: a structural prior with the wrong sign is harmful &
F5 against F1: regret ${f5c['A1']:+.1f}$ (\SIDN) and ${f5c['B0f']:+.1f}$ points (classical); both far ahead of the black-box models & {v4} \\
\bottomrule
\end{{tabular}}
\end{{table}}
"""
    (V2 / "tab_hyp.tex").write_text(tab_hyp, encoding="utf-8")
    check(v1 == "Supported" and v3 == "Supported" and v4 == "Not supported",
          "hypothesis verdicts differ from the Discussion text")

    # the deployment subsection is fixed text; it ships with the code for standalone runs
    static_f = V2 / "discussion_static.tex"
    if not static_f.exists():
        static_f = Path(__file__).resolve().parent / "discussion_static.tex"
    static = static_f.read_text(encoding="utf-8")
    start = static.index(r"\subsection{Deployment")
    end = static.find(r"\section{Limitations", start)
    deploy = static[start:end if end >= 0 else len(static)]
    disc += "\n" + deploy.strip() + "\n"

    pool_lim = ""
    if pooled:
        pool_lim = (f" The pooled variant was evaluated on {K['pool_seeds_inline'].split(',')[0]} only.")
    disc += rf"""
\section{{Limitations and Future Research}}
\label{{sec:limits}}

First, the evaluation is semi-synthetic by necessity: base demand from two
public panels and one synthetic panel is combined with simulated decision
histories and known responses, because the
profit of an unimplemented plan cannot be observed. The five response families
span matched, misspecified, time-varying, and sign-violating cases, but field
data with experimentally varied decisions would be needed to measure how large
the shape discrepancy of Theorem~\ref{{thm:transport}} is in practice. Second,
the guarantees assume that calibration and planning weeks are exchangeable;
trends and structural breaks violate this assumption, and rolling
recalibration mitigated the problem on one panel but not on another. Third,
each panel--family combination was run with {K['seeds_text'] if K['n_seed_min'] == K['n_seed_max'] else 'up to ' + K['seeds_text']}, and the test
periods contain 13--22 weeks; the differences between the two structural
models are not statistically resolved at this scale.{pool_lim}{sens_lim} Fourth, the
sign-violating family perturbs only the circularity response of one segment;
sign errors in the price response were not tested. Fifth, the planner is a local solver; the oracle uses additional
starting points, and for the S-shaped responses of F3 the reported optimum is
the best of several local optima. Finally, the model is single-period and
ignores competitor reactions, inventory dynamics, and the integer decisions of
production scheduling. Future work could estimate the shape discrepancy from
designed experiments, combine the band with adaptive calibration schemes that
retain finite-sample guarantees under drift, and extend the structural prior to
cross-price and dynamic effects.

\section{{Conclusion}}
\label{{sec:conclusion}}

Organizations that let {'machine-learning' if EJOR else 'AI-based'} pipelines set prices, promotions, and recycled
content need to know which accuracy to demand from the forecaster and which
guarantees survive when the optimizer moves decisions away from history. This
paper answered both questions theoretically and tested the answers on a
benchmark with known ground truth. Pricing quality is governed by the accuracy
of the demand response, not by forecast accuracy: the weekly price loss tracked
the elasticity error (Spearman $\rho={K['rho_elast']:.2f}$) and hardly the
hold-out forecast error ($\rho={K['rho_mape']:.2f}$). Distribution-free
guarantees transfer to planned decisions only up to a shape discrepancy that
embedded structure controls: the black-box band lost {f0(gap_b2c)} points of
coverage at its own plans, the structural band {f0(gap_a4)}. Conformal robust
planning turned the transferred guarantee into feasibility, reducing the share
of weeks with a violated limit from up to {f0(100 * max_nom_viol)}\% to at most
{f0(100 * robust_max)}\% at an explicit and adjustable price in expected profit.
Structure, more than flexibility, is what makes {'data-driven' if EJOR else 'AI-based'} plans accurate and
their guarantees meaningful.
"""

    # ------------------------------------------------------------------ abstract and MRS
    abs_robust, mrs_tests = "", ""
    if sf.exists():
        SN = json.loads(sf.read_text(encoding="utf-8"))
        levs = ("0.05", "0.15", "0.3")
        reg_sp = all(SN[f"sp{l}_B2m"]["NR"] > max(SN[f"sp{l}_A1"]["NR"], SN[f"sp{l}_B0f"]["NR"]) for l in levs)
        cov_noise = all(a > b for a, b in zip(SN["cov_A4"], SN["cov_B2c"]))
        if reg_sp and cov_noise and SN.get("cov_gap_all_sp") and SN.get("regret_gap_all_noise"):
            abs_robust = (" Both differences persisted when the historical price variation and the demand"
                          " noise were varied.")
        else:
            flags.append("abstract: robustness sentence dropped (a sensitivity comparison does not hold)")
        if SN["elast_decreasing"]["A1"] and SN["regret_decreasing"]["A1"]:
            mrs_tests = (" Planned price tests also improve the model, because the variation they create"
                         " sharpens the elasticity estimates on which the plans depend.")
    abstract = rf"""Firms increasingly delegate pricing, promotion, and recycled-content
planning to pipelines in which a machine-learning forecaster feeds an optimizer.
Two questions remain open for managers: which forecast accuracy matters for the
quality of a plan, and whether distribution-free guarantees, such as those of
conformal prediction, remain valid when the optimizer moves decisions away from
their historical values. We answer both for weekly planning in a circular
supply chain. We prove that optimal prices are invariant to the level of a
multiplicative demand forecast and derive the closed-form profit loss caused by
an elasticity error. We further prove that a conformal guarantee computed at
historical decisions transfers to planned decisions up to a shape discrepancy
that vanishes when the model's demand response has the true shape, and that the
transferred guarantee yields plan-level feasibility and a profit floor. The
resulting framework, SIDN-CRO, couples a structure-informed demand network with
conformal robust optimization. On a benchmark with known ground truth, built
from three weekly demand panels (two public, one synthetic) and five true
response families, the
structural models forfeited {f0(struct_lo)}--{f0(struct_hi)}\% of the optimal profit
against {f0(black_lo)}--{f0(black_hi)}\% for black-box forecasters; price loss tracked
elasticity error (Spearman {K['rho_elast']:.2f}), not forecast error
({K['rho_mape']:.2f}); and moving from historical to planned decisions cost a
black-box conformal band {f0(gap_b2c)} points of coverage against {f0(gap_a4)} for
SIDN-CRO.{abs_robust} Embedded structure, more than flexibility, makes AI-based plans
accurate and their guarantees meaningful."""
    if EJOR:
        # EJOR: no formulae and no unexplained abbreviations or acronyms in the abstract
        sig = (" (significant at the 0.1\\% level)" if K["p_black_A1"] < 1e-3 else "")
        abstract = rf"""Firms increasingly delegate pricing, promotion, and recycled-content
planning to pipelines in which a machine-learning forecaster feeds an optimizer.
Two questions remain open: which forecast accuracy matters for the quality of a
plan, and whether distribution-free guarantees, such as those of conformal
prediction, remain valid when the optimizer moves decisions away from their
historical values. We answer both for weekly planning in circular supply
chains. We prove that optimal prices are invariant to the level of a
multiplicative demand forecast and derive the closed-form profit loss caused by
an elasticity error. We further prove that a conformal guarantee computed at
historical decisions transfers to planned decisions up to a shape discrepancy
that vanishes when the model's demand response has the true shape, and that the
transferred guarantee yields plan-level feasibility and a profit floor. The
resulting framework couples a structure-informed demand network with conformal
robust optimization. On a known-ground-truth benchmark built from three
weekly demand panels (two public, one synthetic) and five true response
families, the structural models forfeited {f0(struct_lo)}--{f0(struct_hi)}\% of the optimal
profit against {f0(black_lo)}--{f0(black_hi)}\% for black-box forecasters{sig}; price loss
tracked elasticity error (Spearman correlation {K['rho_elast']:.2f}), not forecast
error ({K['rho_mape']:.2f}); and moving from historical to planned decisions cost a
black-box conformal band {f0(gap_b2c)} points of coverage against {f0(gap_a4)} for the
structure-informed band.{abs_robust} Embedded structure, more than flexibility,
makes data-driven plans accurate and their guarantees meaningful."""

    mrs = rf"""Planning teams increasingly let machine-learning models choose prices,
promotions, and recycled-content levels, and they select these models by their
forecast accuracy on past data. This study shows that this practice certifies
the wrong property. The profit of a plan depends on how accurately the model
captures the response of demand to the decisions being planned, and an
uncertainty band that is reliable at past decisions can fail at the decisions
the model recommends: in our benchmark, the band of a black-box forecaster lost
{f0(gap_b2c)} points of coverage on average at its own plans. Managers should therefore audit
the price elasticities that a forecaster implies at the proposed plan rather
than its hold-out error, prefer models with embedded economic structure, whose
uncertainty bands remain informative at new decisions, and choose a
conservatism level that trades expected profit against the risk of violating
capacity and recycled-input commitments.{mrs_tests} The proposed framework makes this
trade-off explicit and auditable and runs on sales, marketing, and procurement
data that planning organizations already hold."""

    (V2 / "results_intro.tex").write_text(intro, encoding="utf-8")
    (V2 / "body_discussion.tex").write_text(disc, encoding="utf-8")
    (V2 / "abstract.tex").write_text(abstract, encoding="utf-8")
    (V2 / "mrs.tex").write_text(mrs, encoding="utf-8")
    wa, wm = words(abstract), words(mrs)
    check(wa <= 250, f"abstract has {wa} words (> 250)")
    check(wm <= 200, f"MRS has {wm} words (> 200)")
    print(f"written: results_intro, body_discussion, abstract ({wa} words), mrs ({wm} words)")
    print("FLAGS:" if flags else "no flags", *flags, sep="\n  ")


if __name__ == "__main__":
    main()
