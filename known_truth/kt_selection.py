# -*- coding: utf-8 -*-
"""
Model selection in practice: which forecaster would a planner choose, and what would it cost?

Uses only the stored result files (no new runs). In every run (panel, truth family, seed) the
test weeks are split chronologically: in the first half (selection weeks) the planner chooses one
of the nominal models by a selection rule; in the second half (evaluation weeks) the normalized
regret of the chosen model's plans is recorded. Rules:
  forecast  - lowest forecast error (MAPE) at the historical decisions of the selection weeks
              (what common practice observes);
  struct    - the same criterion, restricted to the two structural models (B0f, A1);
  plan      - highest profit of the model's plans in the selection weeks, evaluated under the
              true expected demand (an idealized pilot of the candidate plans);
  hindsight - lowest regret in the evaluation weeks themselves (unattainable reference).
Writes tab_selection.tex, body_selection.tex, and selection_numbers.json into KT_PAPER_DIR.
Every qualitative statement is checked and reported as a FLAG if it does not hold.
"""
import json
import os
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results" / "known_truth"
V2 = Path(os.environ.get("KT_PAPER_DIR", str(HERE.parent / "results" / "paper")))
(V2 / "figs").mkdir(parents=True, exist_ok=True)   # output folder of the tables, figures, and text
NOMINAL = ["B0f", "B1", "B2", "B2m", "B4", "A1"]
BLACK = {"B1", "B2", "B2m", "B4"}
RULES = ["forecast", "struct", "plan", "hindsight"]
flags = []


def check(cond, msg):
    if not cond:
        flags.append(msg)
    return cond


def boot_mean_ci(d, reps=10000, seed=2026):
    d = np.asarray(d, float)
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), size=(reps, len(d)))].mean(axis=1)
    return [float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def runs():
    for f in sorted(RES.glob("*_seed*.json")):
        if f.name.startswith("pilot"):
            continue
        d = json.loads(f.read_text())
        if "models" in d:                      # skip placeholders of deferred tasks
            yield d


def main():
    rec = []
    for d in runs():
        M = d["models"]
        T = len(M["A1"]["NR"])
        h = T // 2
        sel, ev = slice(0, h), slice(h, T)

        def mean(model, key, s):
            return float(np.nanmean(np.asarray(M[model][key], float)[s]))

        pick = {"forecast": min(NOMINAL, key=lambda m: mean(m, "mape", sel)),
                "struct": min(["B0f", "A1"], key=lambda m: mean(m, "mape", sel)),
                "plan": max(NOMINAL, key=lambda m: mean(m, "profit_true", sel)),
                "hindsight": min(NOMINAL, key=lambda m: mean(m, "NR", ev))}
        rec.append({"cell": (d["dataset"], d["family"]), "seed": d["seed"], "pick": pick,
                    "nr": {r: mean(pick[r], "NR", ev) for r in RULES}, "n_sel": h, "n_ev": T - h})

    cells = sorted({r["cell"] for r in rec})
    n_runs = len(rec)
    # equal weight per panel--family cell, as in the main results
    cell_nr = {r: float(np.mean([np.mean([x["nr"][r] for x in rec if x["cell"] == c]) for c in cells]))
               for r in RULES}
    picks_black = {r: sum(x["pick"][r] in BLACK for x in rec) for r in RULES}
    picks_a1 = {r: sum(x["pick"][r] == "A1" for x in rec) for r in RULES}
    diff = [x["nr"]["forecast"] - x["nr"]["plan"] for x in rec]
    d_ci = boot_mean_ci(diff)
    differ = sum(x["pick"]["forecast"] != x["pick"]["plan"] for x in rec)
    w = stats.wilcoxon([x["nr"]["forecast"] for x in rec], [x["nr"]["plan"] for x in rec])
    worse_black = [x["nr"]["forecast"] for x in rec if x["pick"]["forecast"] in BLACK]
    gap_plan = cell_nr["plan"] - cell_nr["hindsight"]

    check(cell_nr["forecast"] > cell_nr["plan"], "forecast-based selection is not worse than plan-based")
    check(d_ci[1] > 0, "the CI of the forecast-minus-plan difference includes zero")
    check(w.pvalue < 0.05, "the Wilcoxon test of forecast vs plan selection is not significant")
    check(picks_black["plan"] == 0, "plan-based selection chose a black-box model")
    check(picks_black["forecast"] > 0, "forecast-based selection never chose a black-box model")
    check(cell_nr["struct"] < cell_nr["forecast"], "restricting to structural models does not help")

    S = {"n_runs": n_runs, "n_cells": len(cells), "cell_nr": cell_nr, "picks_black": picks_black,
         "picks_a1": picks_a1, "diff_forecast_plan": d_ci, "p_wilcoxon": float(w.pvalue),
         "n_differ": differ, "nr_when_black": float(np.mean(worse_black)) if worse_black else None,
         "n_sel_range": [min(x["n_sel"] for x in rec), max(x["n_sel"] for x in rec)],
         "n_ev_range": [min(x["n_ev"] for x in rec), max(x["n_ev"] for x in rec)]}

    f1 = lambda x: f"{x:.1f}"  # noqa: E731
    p_txt = "$p<0.001$" if w.pvalue < 1e-3 else f"$p={w.pvalue:.3f}$"
    names = {"forecast": "Forecast error, all models",
             "struct": "Forecast error, structural only",
             "plan": "Plan profit in a pilot",
             "hindsight": "Best in hindsight (reference)"}
    tab = r"""\begin{table}[!t]
\caption{Model selection by a planner: normalized regret (\%) in the evaluation weeks of the model
chosen in the selection weeks (first half of the test weeks of each run), equal weight per
panel--family cell, and how often the rule chose a black-box model or \SIDN (""" + str(n_runs) + r""" runs).}
\label{tab:selection}
\centering\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}lccc@{}}
\toprule
Selection rule & Regret & Black box & \SIDN \\
\midrule
""" + "".join(f"{names[r]} & {f1(cell_nr[r])} & {picks_black[r]} & {picks_a1[r]} \\\\\n" for r in RULES) + r"""\bottomrule
\end{tabular}
\end{table}
"""
    text = rf"""\subsection{{Selecting a Forecaster for Planning}}
\label{{ssec:res_selection}}

The preceding results compare models whose quality is known. A planner, however, must choose a
forecaster before its plans can be judged, and common practice chooses by forecast accuracy. We
replay this choice in every run: in the first half of the test weeks
({S['n_sel_range'][0]}--{S['n_sel_range'][1]} weeks) the planner chooses one of the six nominal
models, and in the remaining weeks we record the regret of the chosen model's plans
(Table~\ref{{tab:selection}}). Choosing the model with the lowest forecast error yields a regret of
{f1(cell_nr['forecast'])}\%; choosing the model whose plans earned the highest profit in the
selection weeks yields {f1(cell_nr['plan'])}\%, within {f1(gap_plan)} points of the best model in
hindsight ({f1(cell_nr['hindsight'])}\%). The difference between the two rules,
${f1(d_ci[0])}$ points (95\% CI ${f1(d_ci[1])}$ to ${f1(d_ci[2])}$; paired Wilcoxon test, {p_txt}),
arises in the {differ} of {n_runs} runs in which the rules disagree. Forecast accuracy chose a
black-box model in {picks_black['forecast']} runs, whose plans then forfeited
{f1(S['nr_when_black'])}\% of the optimal profit on average, whereas plan profit never did.
Restricting the choice to the two structural models already lowers the regret of
accuracy-based selection to {f1(cell_nr['struct'])}\%, but it remains above that of plan-based
selection, because forecast accuracy also fails to rank the two structural models. The pilot is
idealized: it evaluates each candidate's plans under the expected demand, whereas a real pilot
observes noisy realized profit and tests one plan at a time. The comparison nevertheless shows
that the selection criterion, not only the model class, determines the quality of the plans.
"""
    (V2 / "tab_selection.tex").write_text(tab, encoding="utf-8")
    (V2 / "body_selection.tex").write_text(text, encoding="utf-8")
    (V2 / "selection_numbers.json").write_text(json.dumps(S, indent=1), encoding="utf-8")
    print(json.dumps({k: S[k] for k in ("n_runs", "cell_nr", "picks_black", "diff_forecast_plan",
                                         "p_wilcoxon", "n_differ")}, indent=1))
    print("FLAGS:" if flags else "no flags", *flags, sep="\n  ")


if __name__ == "__main__":
    main()
