"""
Aggregate the known-truth benchmark: tables and hypothesis tests.

Outputs (results/known_truth/summary/):
    table_nr.csv          mean (sd over seeds) normalised regret by dataset x family x model
    table_ops.csv         realised violation rate, coverage at planned decisions,
                          elasticity error, MAPE
    tests_nr.csv          paired Wilcoxon tests of A4 and A1 against each model
                          (seed-level means; Holm correction within dataset x family)
    h1_regression.csv     week-level regressions of price loss and NR on elasticity
                          error and MAPE (standardised)
    summary.json          everything above in one file
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results" / "known_truth"
OUT = RES / "summary"
MODELS = ["B0f", "B1", "B2", "B2m", "B4", "A1", "A4", "A4r", "A4_75", "A4_50", "A4_25", "B2c",
          "A1p", "A4p", "A4rp", "A4p_50"]
POOL = HERE.parent / "results" / "known_truth_pool"


def pool_complete_seeds(n_cells=15):
    """Seeds for which the SIDN-P variant was run in every panel--family cell."""
    cnt = {}
    for f in POOL.glob("*_seed*.json"):
        s = int(f.stem.split("_seed")[1])
        cnt[s] = cnt.get(s, 0) + 1
    return sorted(s for s, c in cnt.items() if c >= n_cells)


def load():
    rows, weeks = [], []
    pool_seeds = set(pool_complete_seeds())
    for f in sorted(RES.glob("*_seed*.json")):
        if f.name.startswith("pilot"):
            continue
        d = json.loads(f.read_text())
        if "models" not in d:                 # placeholder for a task deferred to a later round
            continue
        fp = POOL / f.name
        if fp.exists() and d["seed"] in pool_seeds:   # merge the SIDN-P variant (complete seeds only)
            dp = json.loads(fp.read_text())
            assert np.allclose(dp["oracle"]["pi_star"], d["oracle"]["pi_star"], rtol=1e-6), f.name
            d["models"].update(dp["models"])
        for m in MODELS:
            if m not in d["models"]:
                continue
            r = d["models"][m]
            cov = [c for c in r["cover"] if c is not None]
            covh = [c for c in r.get("cover_hist", []) if c is not None]
            rows.append({"dataset": d["dataset"], "family": d["family"], "seed": d["seed"], "model": m,
                         "NR": np.mean(r["NR"]), "viol": np.mean(r["viol"]),
                         "cover": np.mean(cov) if cov else np.nan,
                         "cover_hist": np.mean(covh) if covh else np.nan,
                         "dist_p": np.mean(r["dist_p"]) if r.get("dist_p") else np.nan,
                         "elast_err": np.mean(r["elast_err"]), "elast_bias": np.mean(r["elast_bias"]),
                         "mape": np.mean(r["mape"]), "price_loss": np.mean(r["price_loss"]),
                         "feas_planned": np.mean(r["feas_planned"])})
            for w in range(len(r["NR"])):
                weeks.append({"dataset": d["dataset"], "family": d["family"], "seed": d["seed"],
                              "model": m, "week": w, "NR": r["NR"][w], "price_loss": r["price_loss"][w],
                              "elast_err": r["elast_err"][w], "mape": r["mape"][w],
                              "viol": r["viol"][w], "oracle_viol": d["oracle"]["viol"][w]})
    return pd.DataFrame(rows), pd.DataFrame(weeks)


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, k in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[k]))
        adj[k] = running
    return adj


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    runs, weeks = load()
    if runs.empty:
        print("no results yet")
        return
    g = runs.groupby(["dataset", "family", "model"])
    table = g.agg(NR_mean=("NR", "mean"), NR_sd=("NR", "std"), n_seeds=("seed", "nunique"),
                  viol=("viol", "mean"), cover=("cover", "mean"), cover_hist=("cover_hist", "mean"),
                  dist_p=("dist_p", "mean"), elast_err=("elast_err", "mean"),
                  elast_bias=("elast_bias", "mean"), mape=("mape", "mean"),
                  price_loss=("price_loss", "mean")).reset_index()
    table.to_csv(OUT / "table_all.csv", index=False)

    tests = []
    for (ds, fam), sub in runs.groupby(["dataset", "family"]):
        piv = sub.pivot(index="seed", columns="model", values="NR")
        for ref in ("A4", "A1"):
            ps, rec = [], []
            for m in MODELS:
                if m == ref or m not in piv or ref not in piv:
                    continue
                pair = piv[[ref, m]].dropna()
                if len(pair) < 3 or np.allclose(pair[ref], pair[m]):
                    p = 1.0
                else:
                    p = stats.wilcoxon(pair[ref], pair[m]).pvalue
                ps.append(p)
                rec.append({"dataset": ds, "family": fam, "ref": ref, "other": m, "n": len(pair),
                            "diff_mean": float((pair[ref] - pair[m]).mean()),
                            "ref_better_share": float((pair[ref] < pair[m]).mean()), "p": p})
            for r_, pa in zip(rec, holm(ps)):
                r_["p_holm"] = pa
                tests.append(r_)
    tests = pd.DataFrame(tests)
    tests.to_csv(OUT / "tests_nr.csv", index=False)

    # H1: price loss and NR on standardised elasticity error and MAPE (week level)
    h1 = []
    for (ds, fam), sub in weeks.groupby(["dataset", "family"]):
        s = sub.replace([np.inf, -np.inf], np.nan).dropna(subset=["elast_err", "mape", "price_loss", "NR"])
        if len(s) < 20:
            continue
        Z = np.column_stack([np.ones(len(s)),
                             (s.elast_err - s.elast_err.mean()) / s.elast_err.std(),
                             (s.mape - s.mape.mean()) / s.mape.std()])
        for target in ("price_loss", "NR"):
            y = s[target].values
            beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
            rho_e = stats.spearmanr(s.elast_err, y).correlation
            rho_m = stats.spearmanr(s.mape, y).correlation
            h1.append({"dataset": ds, "family": fam, "target": target, "n": len(s),
                       "coef_elast": beta[1], "coef_mape": beta[2],
                       "spearman_elast": rho_e, "spearman_mape": rho_m})
    h1 = pd.DataFrame(h1)
    h1.to_csv(OUT / "h1_regression.csv", index=False)

    summary = {"table": table.to_dict(orient="records"), "tests": tests.to_dict(orient="records"),
               "h1": h1.to_dict(orient="records"),
               "oracle_viol": weeks.groupby(["dataset", "family"])["oracle_viol"].mean().to_dict().__repr__()}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    pd.set_option("display.width", 200)
    piv = table.pivot_table(index=["dataset", "family"], columns="model", values="NR_mean").round(2)
    print(piv[[m for m in MODELS if m in piv.columns]].to_string())
    print()
    piv = table.pivot_table(index=["dataset", "family"], columns="model", values="viol").round(2)
    print(piv[[m for m in MODELS if m in piv.columns]].to_string())
    print()
    print(h1.round(3).to_string())


if __name__ == "__main__":
    main()
