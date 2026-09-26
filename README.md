# SIDN-CRO — Code, Data, and Results

Code and reproducibility material for the paper:

> **"Decision-Relevant Accuracy and Transportable Guarantees for AI-Based
> Planning in Circular Supply Chains"**
> Qazi Salman Khalid, Muzzamil Mustafa, Usman Aftab Butt, Tajdar Khan.

The repository reproduces every experiment, table, figure, and number of the
paper and its Supplementary Material. The experiments use a **known-ground-truth
benchmark**: base demand from three weekly panels (DataCo and Olist, public;
Synth-2026, synthetic) is combined with five true demand-response families, so
that the true profit of any plan and the true optimum can be computed.

---

## 1. Layout

| Path | Content |
|---|---|
| `known_truth/` | The known-ground-truth benchmark and every script that generates the reported numbers (Section 2) |
| `src/` | Demand models (`src/models/`: SIDN, MLP, XGBoost, BiLSTM with attention), dataset loading (`src/train/dataset.py`), and the pipeline that built the preprocessed panels and the calibrated advertising and recycled-content overlays (`src/data_pipeline/`) |
| `data/` | Preprocessed weekly panels: DataCo (`Preprocessed/`), Olist (`Preprocessed_Olist/`), Synth-2026 (`Preprocessed_Synth2026/`); `Raw_Data/README_sources.md` documents the public sources |
| `results/known_truth/` | One JSON file per run (`{panel}_{family}_seed{s}.json`, seeds 42–46), the calibrated operational limits (`econ_{panel}.json`), run logs, and summaries used by the paper (`summary/`) |
| `results/known_truth_pool/` | The pooled variant SIDN-P (Supplementary Section S3), seeds 42–43 |
| `results/known_truth_sens/` | Sensitivity runs: historical price variation (`sp0.05/`, `sp0.30/`) and demand noise (`sd0.05/`, `sd0.20/`), seeds 42–44 |
| `code/`, `logs/`, `results/*.json`, `results/*.csv` | Material of an earlier version of the study (Section 6) |

## 2. The benchmark (`known_truth/`)

| File | Role |
|---|---|
| `kt_truth.py` | The five true demand-response families F1–F5 (shared parameters, seed 2026) |
| `kt_data.py` | Builds a benchmark instance: base demand, simulated prices, historical overlays, realized demand |
| `kt_econ.py`, `kt_calib.py` | Planning economics; calibration of capacity and recycled-input limits on the validation weeks |
| `kt_solver.py` | SQP planner on normalized variables (oracle and models) |
| `kt_models.py` | Model fitting (B0f, B1, B2, B2m, B4, A1) and the conformal layer (static, rolling, scaled) |
| `kt_pool.py` | SIDN-P |
| `kt_run.py`, `kt_run_pool.py`, `kt_run_sens.py` | Runners: one task per (panel, family, seed) |
| `kt_analyze.py` | Loads all result files |
| `kt_tables.py` | Tables and figures of the Results section and the supplementary pooled-variant table |
| `kt_results_text.py`, `kt_sens_text.py`, `kt_discussion_text.py`, `kt_supplement_text.py` | Generate the result-dependent text of the paper (Results, sensitivity analysis, Introduction summary, Discussion, Limitations, Conclusion, abstract, managerial relevance statement, Supplementary Sections S3–S4). Every number is computed; every qualitative statement is checked against the data and reported as a FLAG if it does not hold |
| `fig_mechanism.py`, `fig_theory.py` | Mechanism figure (one instance) and the loss curves of Proposition 3 |
| `diag_confound.py` | Correlation of historical advertising and recycled content with base demand (Supplementary Table S2) |
| `time_deploy.py` | Wall-clock time of SIDN training, conformal calibration, and one robust plan on one CPU thread |
| `test_oracle.py`, `test_calib.py` | Checks of the oracle against the closed-form markup, of convergence from different starting points, and of how often the limits bind |
| `smoke.py`, `smoke_pool.py`, `smoke_sens.py`, `pilot.py` | Quick end-to-end checks on one instance |

## 3. Environment

Python 3.11; CPU only. Versions used: numpy 2.4, scipy 1.17, scikit-learn 1.8,
torch 2.11, xgboost 3.2, pandas 2.3, matplotlib 3.10.

```bash
pip install -r requirements.txt
```

## 4. Reproducing the results

Run from the repository root.

```bash
# 1. main benchmark: 3 panels x 5 truth families x seeds (one JSON per run)
python known_truth/kt_run.py --workers 14 --seeds 42 43 44 45 46

# 2. pooled variant SIDN-P and sensitivity runs
python known_truth/kt_run_pool.py --workers 14 --seeds 42 43
python known_truth/kt_run_sens.py --workers 14 --seeds 42 43 44

# 3. diagnostics and single-instance figures
python known_truth/diag_confound.py
python known_truth/fig_mechanism.py dataco
python known_truth/fig_theory.py
python known_truth/time_deploy.py

# 4. tables, figures, and all result-dependent text (written to KT_PAPER_DIR, default results/paper)
python known_truth/kt_tables.py
python known_truth/kt_results_text.py
python known_truth/kt_sens_text.py
python known_truth/kt_discussion_text.py
python known_truth/kt_supplement_text.py
```

The runners skip tasks whose result file exists, so steps 1 and 2 can be
interrupted and resumed. With 16 concurrent runs on a 10-core laptop CPU, one
run of the main benchmark takes 1–3 hours (median about 2 hours). The result
files of all runs reported in the paper are included, so step 4 can be run
directly.

**Result files.** Each `{panel}_{family}_seed{s}.json` contains the oracle
profit and violation per test week and, for every model, per test week:
normalized regret (`NR`), violation of a demand-dependent limit under realized
demand (`viol`), coverage of the conformal band at the planned decision
(`cover`) and at the historical decisions (`cover_hist`), mean absolute and
signed elasticity error at the plan (`elast_err`, `elast_bias`), forecast error
at the historical decisions (`mape`), price loss (`price_loss`), distance of the
planned prices from the historical prices (`dist_p`), and the realized profit
(`profit_true`).

## 5. Datasets

| Dataset | Source | License |
|---|---|---|
| **DataCo Supply Chain** | Mendeley Data, V3, DOI [10.17632/8gx2fvg2k6.3](https://doi.org/10.17632/8gx2fvg2k6.3) | CC BY 4.0 |
| **Olist Brazilian E-Commerce** | [Kaggle: olistbr/brazilian-ecommerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) | CC BY-NC-SA 4.0 |
| **Synth-2026** | Synthetic; regenerated by `src/data_pipeline/build_synth2026.py` (fixed seeds) | — |

None of the public datasets records cooperative-advertising spend or
recycled-content shares; these histories are calibrated overlays generated by
`src/data_pipeline/synth_ad_calibration*.py` and
`src/data_pipeline/synth_circular_calibration*.py` (Supplementary Section S5).
The preprocessed panels in `data/` are what the benchmark consumes; the raw
archives are not redistributed (see `data/Raw_Data/README_sources.md`).

## 6. Earlier version of the study

The folders `code/` and `logs/` and the files `results/*.json` and
`results/*.csv` belong to an earlier version of this study, which evaluated the
models with a different protocol. They are superseded by the known-ground-truth
benchmark, are not used in the current paper, and are kept for transparency.
The H&M panel of that version is not redistributed (Kaggle competition rules)
and is regenerated by `src/data_pipeline/ingest_hm.py`.

## License

MIT (see `LICENSE`).
