"""
SIDN-CRO -- Statistical significance tests (Wilcoxon signed-rank).

Compares A4 (CRODemand) against all baseline models (B0-B5, A1, A2) using
the non-parametric Wilcoxon signed-rank test on per-seed mean NR% values.

Rationale for test choice
--------------------------
The Wilcoxon signed-rank test is the appropriate non-parametric counterpart
of the paired t-test.  It is preferred here because:
  1. N_seeds = 10 is small: the CLT for the paired t-test is not reliable.
  2. NR% values may be heavy-tailed (driven by irregular high-demand weeks).
  3. Observations are paired: same (dataset, week, seed) for A4 and each
     baseline, so intra-seed correlation must be accounted for.

Test procedure
--------------
For each (dataset, baseline) pair:
  1. Compute per-seed mean NR% for A4 and for the baseline.
  2. Run Wilcoxon signed-rank test on the N_seeds paired differences.
  3. Report two-sided p-value, Bonferroni-adjusted p-value (K comparisons
     per dataset, K = number of non-missing baselines), and rank-biserial
     correlation r as the effect size.

Bonferroni correction is applied per dataset (not globally), which is
the standard approach in ML comparison papers.

Usage
-----
    # After running experiments:
    python experiments/statistical_tests.py --results_dir results/

    # Specific dataset and target model:
    python experiments/statistical_tests.py --datasets dataco olist --target A4

    # Print LaTeX table for manuscript:
    python experiments/statistical_tests.py --latex

    # Save results to JSON:
    python experiments/statistical_tests.py --output results/sig_tests.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy import stats as scipy_stats

# Force UTF-8 output on Windows cp1252 consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.paths import RESULTS_DIR


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
# Full model order: baselines to test against TARGET_MODEL
BASELINE_ORDER = ["B0", "B1", "B2", "B3", "B4", "B5", "A1", "A2"]
TARGET_MODEL   = "A4"
ALPHA_NOMINAL  = 0.05       # nominal significance level (Bonferroni-adjusted per dataset)


# --------------------------------------------------------------------------- #
# Result loading
# --------------------------------------------------------------------------- #
def load_seed_nr(
    results_dir: Path,
    dataset_key: str,
    models: List[str],
) -> Dict[str, Dict[int, float]]:
    """Load per-seed mean NR% for each model from saved JSON result files.

    Parameters
    ----------
    results_dir : root results directory (contains raw/ subfolder)
    dataset_key : e.g. "dataco", "olist", "hm", "synth"
    models      : list of model names to load

    Returns
    -------
    { model_name: { seed: mean_NR_pct } }
    """
    raw_dir = results_dir / "raw"
    out: Dict[str, Dict[int, float]] = {m: {} for m in models}

    for path in sorted(raw_dir.glob(f"{dataset_key}_*.json")):
        try:
            with open(path) as f:
                rec = json.load(f)
        except Exception:
            continue
        mn   = rec.get("model", "")
        seed = rec.get("seed", -1)
        if mn not in models or seed < 0:
            continue

        # Primary: NR_mean from summary (already in %)
        summary  = rec.get("summary", {})
        nr_mean  = summary.get("NR_mean", None)

        # Fallback: compute from per-week regret_norm
        if nr_mean is None:
            wr = rec.get("week_results", [])
            if wr:
                nr_mean = float(
                    np.mean([w["regret_norm"] for w in wr]) * 100.0)
            else:
                continue

        out[mn][seed] = float(nr_mean)

    return out


# --------------------------------------------------------------------------- #
# Wilcoxon signed-rank test (paired)
# --------------------------------------------------------------------------- #
def wilcoxon_test(
    a_values: np.ndarray,    # A4 NR% per seed  (lower = better)
    b_values: np.ndarray,    # Baseline NR% per seed
) -> Dict[str, float]:
    """Paired Wilcoxon signed-rank test.

    H0: median(A4_NR - Baseline_NR) = 0  (two-sided).

    A negative median_diff means A4 achieves lower NR% (is better).

    Parameters
    ----------
    a_values : (N,) NR% for the target model (A4) per seed
    b_values : (N,) NR% for the baseline per seed (same seed order)

    Returns
    -------
    dict with keys:
        statistic   : Wilcoxon W statistic
        p_value     : two-sided p-value
        effect_r    : rank-biserial correlation (effect size, in [-1, 1])
        median_diff : median(a - b); negative = A4 better
        n           : number of paired observations (seeds)
    """
    if len(a_values) < 2 or len(b_values) < 2:
        return {"statistic": np.nan, "p_value": np.nan,
                "effect_r": np.nan, "median_diff": np.nan, "n": 0}

    diff = a_values - b_values      # negative = A4 better
    n    = len(diff)

    # Scipy Wilcoxon (drops zero differences internally via zero_method="wilcox")
    stat, p_val = scipy_stats.wilcoxon(
        a_values, b_values,
        alternative="two-sided",
        zero_method="wilcox",
    )

    # Effect size: rank-biserial correlation r = Z / sqrt(N_nonzero)
    # where Z is the normal approximation of the signed-rank statistic.
    nz   = diff[diff != 0]
    n_nz = len(nz)
    if n_nz >= 2:
        e_w   = n_nz * (n_nz + 1) / 4.0
        var_w = n_nz * (n_nz + 1) * (2 * n_nz + 1) / 24.0
        z     = (stat - e_w) / (np.sqrt(var_w) + 1e-12)
        r     = float(z / np.sqrt(n_nz))
    else:
        r = 0.0

    return {
        "statistic":   float(stat),
        "p_value":     float(p_val),
        "effect_r":    r,
        "median_diff": float(np.median(diff)),
        "n":           int(n),
    }


# --------------------------------------------------------------------------- #
# Run all pairwise comparisons for one dataset
# --------------------------------------------------------------------------- #
def run_tests_for_dataset(
    results_dir: Path,
    dataset_key: str,
    target:    str = TARGET_MODEL,
    baselines: List[str] = None,
    verbose:   bool = True,
) -> Dict[str, Dict[str, float]]:
    """Run Wilcoxon tests for target vs each baseline on one dataset.

    Parameters
    ----------
    results_dir : path to results root (with raw/ subfolder)
    dataset_key : dataset identifier key
    target      : model name to treat as the proposed method
    baselines   : list of baseline model names to compare against
    verbose     : print table to stdout

    Returns
    -------
    { baseline_name: { statistic, p_value, p_adjusted, effect_r,
                       median_diff, significant, n,
                       a4_mean_nr, bl_mean_nr, improvement } }
    """
    if baselines is None:
        baselines = BASELINE_ORDER

    all_models   = list(dict.fromkeys(baselines + [target]))
    seed_results = load_seed_nr(results_dir, dataset_key, all_models)

    target_seeds = seed_results.get(target, {})
    if not target_seeds:
        print(f"  [WARN] No {target} results found for dataset={dataset_key}")
        return {}

    # Bonferroni correction: one family per dataset
    K       = sum(1 for b in baselines if seed_results.get(b))
    K       = max(K, 1)
    alpha_b = ALPHA_NOMINAL / K

    results: Dict[str, Dict[str, float]] = {}
    for bl in baselines:
        bl_seeds = seed_results.get(bl, {})
        if not bl_seeds:
            continue

        # Align on shared seeds (intersection)
        shared = sorted(set(target_seeds) & set(bl_seeds))
        if len(shared) < 3:
            print(f"  [WARN] {target} vs {bl} on {dataset_key}: "
                  f"only {len(shared)} shared seeds (need >= 3)")
            continue

        a_vals = np.array([target_seeds[s] for s in shared])
        b_vals = np.array([bl_seeds[s]     for s in shared])
        test   = wilcoxon_test(a_vals, b_vals)

        # Bonferroni-adjusted p-value (capped at 1)
        p_adj  = min(1.0, test["p_value"] * K)
        sig    = p_adj < ALPHA_NOMINAL

        results[bl] = {
            **test,
            "p_adjusted":  p_adj,
            "significant": sig,
            "a4_mean_nr":  float(a_vals.mean()),
            "bl_mean_nr":  float(b_vals.mean()),
            # positive = A4 has lower NR% (better)
            "improvement": float(b_vals.mean() - a_vals.mean()),
        }

    if verbose:
        _print_test_table(results, dataset_key, target, alpha_b, K)

    return results


# --------------------------------------------------------------------------- #
# Display helpers
# --------------------------------------------------------------------------- #
def _print_test_table(
    results:     Dict,
    dataset_key: str,
    target:      str,
    alpha_adj:   float,
    K:           int,
) -> None:
    """Print a formatted Wilcoxon significance table to stdout."""
    print(f"\n{'='*80}")
    print(f"  Wilcoxon signed-rank: {target} vs baselines | dataset={dataset_key}")
    print(f"  Bonferroni: K={K}  alpha_adj = 0.05/{K} = {alpha_adj:.4f}")
    print(f"{'='*80}")
    hdr = (f"  {'Baseline':<8} {'NR%(A4)':>9} {'NR%(BL)':>9} "
           f"{'Improve':>9} {'W':>8} {'p-val':>9} "
           f"{'p_adj':>9} {'r':>7} {'Sig?':>6}")
    print(hdr)
    print(f"  {'-'*77}")
    for bl, res in results.items():
        star = "**" if res["significant"] else "  "
        print(f"  {bl:<8} "
              f"{res['a4_mean_nr']:>9.2f} "
              f"{res['bl_mean_nr']:>9.2f} "
              f"{res['improvement']:>+9.2f} "
              f"{res['statistic']:>8.1f} "
              f"{res['p_value']:>9.4f} "
              f"{res['p_adjusted']:>9.4f} "
              f"{res['effect_r']:>7.3f} "
              f"{star:>6}")
    print(f"\n  ** = significant at Bonferroni-adjusted alpha = {alpha_adj:.4f}")
    print(f"  Improve = NR%(BL) - NR%(A4);  positive = A4 has lower regret")


def print_latex_table(
    all_results: Dict[str, Dict[str, Dict]],
    target:      str = TARGET_MODEL,
) -> None:
    """Print a LaTeX tabular for the significance appendix (all datasets).

    One row per baseline; columns: NR%(A4), NR%(BL), p_adj for each dataset.
    """
    datasets  = list(all_results.keys())
    baselines = BASELINE_ORDER
    n_ds      = len(datasets)

    col_spec  = "l" + "rrr" * n_ds
    tier_hdr  = " & ".join(
        [f"\\multicolumn{{3}}{{c}}{{{ds.upper()}}}" for ds in datasets])
    col_hdr   = " & ".join(
        [r"$\overline{\mathrm{NR}}_{\mathrm{A4}}$ & "
         r"$\overline{\mathrm{NR}}_{\mathrm{BL}}$ & $p_{\mathrm{adj}}$"]
        * n_ds)

    print()
    print(r"\begin{tabular}{" + col_spec + "}")
    print(r"\toprule")
    print(f"Baseline & {tier_hdr} \\\\")
    # cmidrule for each dataset block (cols 2-4, 5-7, ...)
    rules = "".join(
        [f"\\cmidrule(lr){{{2 + 3*i}-{4 + 3*i}}}"
         for i in range(n_ds)])
    print(r"\cmidrule(r){1-1}" + rules)
    print(f"& {col_hdr} \\\\")
    print(r"\midrule")

    for bl in baselines:
        row = [bl]
        any_data = False
        for ds in datasets:
            res = all_results.get(ds, {}).get(bl)
            if res is None:
                row.append("--- & --- & ---")
            else:
                any_data = True
                star = r"$^{**}$" if res.get("significant") else ""
                row.append(
                    f"{res['a4_mean_nr']:.2f} & "
                    f"{res['bl_mean_nr']:.2f} & "
                    f"{res['p_adjusted']:.4f}{star}"
                )
        if any_data:
            print(" & ".join(row) + r" \\")

    print(r"\bottomrule")
    print(r"\end{tabular}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="Wilcoxon signed-rank significance tests for SIDN-CRO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--results_dir", type=Path, default=RESULTS_DIR,
        help="Root directory with raw/ subfolder of JSON results",
    )
    parser.add_argument(
        "--datasets", nargs="+",
        default=["dataco", "olist", "hm", "synth"],
        help="Datasets to analyse (default: all 4)",
    )
    parser.add_argument(
        "--target", default=TARGET_MODEL,
        help=f"Proposed model to compare against baselines (default: {TARGET_MODEL})",
    )
    parser.add_argument(
        "--baselines", nargs="+", default=BASELINE_ORDER,
        help="Baseline model names to compare against (default: all in BASELINE_ORDER)",
    )
    parser.add_argument(
        "--latex", action="store_true",
        help="Also print LaTeX tabular code suitable for manuscript appendix",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Save full results to this JSON path",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    raw_dir     = results_dir / "raw"
    if not raw_dir.exists():
        print(f"[ERROR] Raw results directory not found: {raw_dir}")
        print("  Run 'python experiments/run_all.py' first to generate results.")
        sys.exit(1)

    all_results: Dict[str, Dict] = {}
    for ds in args.datasets:
        print(f"\n>>> Dataset: {ds}")
        res = run_tests_for_dataset(
            results_dir, ds,
            target=args.target,
            baselines=args.baselines,
            verbose=True,
        )
        all_results[ds] = res

    if args.latex:
        print("\n\n" + "=" * 80)
        print("  LaTeX table (paste into manuscript appendix)")
        print("=" * 80)
        print_latex_table(all_results, target=args.target)

    if args.output:
        # Make all values JSON-serialisable
        def _ser(v):
            if isinstance(v, (np.floating, float)):
                return float(v)
            if isinstance(v, (np.integer, int)):
                return int(v)
            if isinstance(v, (np.bool_, bool)):
                return bool(v)
            if isinstance(v, dict):
                return {k2: _ser(v2) for k2, v2 in v.items()}
            return v

        out = {ds: {bl: {k: _ser(val) for k, val in r.items()}
                    for bl, r in res.items()}
               for ds, res in all_results.items()}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
