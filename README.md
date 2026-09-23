# SIDN-CRO — Code, Data, and Results

Code and reproducibility material for the paper:

> **"A Structure-Informed Predict-then-Optimize Framework with Conformal
> Guarantees for Circular Supply Chain Management"**
> Qazi Salman Khalid, Muzzamil Mustafa, Usman Aftab Butt, Tajdar Khan.

This repository contains **all code, results, and execution logs** needed to
reproduce every table and figure in the paper, plus the preprocessed weekly
panels for DataCo, Olist, and Synth-2026. The H&M panel is not redistributed
(see Section 4) but is regenerated from the public source by the included
ingestion script.

---

## 1. Folder structure

```
Supplementary Material and Code/
├── README.md                  This file.
├── requirements.txt           pip dependencies.
├── environment.yml            conda environment (recommended).
│
├── code/                      Experiment entry points
│   ├── run_all.py             Trains/evaluates all 9 models × 4 datasets × 10 seeds.
│   ├── sensitivity_delta.py   Budget-scale sensitivity analysis (A4).
│   ├── statistical_tests.py   Wilcoxon signed-rank tests (Bonferroni-corrected).
│   └── fast_run.py            Single-seed runner for quick verification.
│
├── src/                       Source package (imported by code/)
│   ├── models/                SIDN, baselines, conformal (CRO), DRO, recurrent nets.
│   ├── optim/                 SQP decision layer + differentiable optimisation.
│   ├── train/                 Dataset loaders and training loops.
│   ├── eval/                  Metrics, feasibility, sensitivity.
│   ├── data_pipeline/         Raw→panel ingestion + synthetic calibration scripts.
│   └── utils/                 Path helpers, plotting.
│
├── data/
│   ├── Preprocessed/          DataCo weekly panel (Train/Validation/Test).
│   ├── Preprocessed_Olist/    Olist weekly panel.
│   ├── Preprocessed_HM/        (not included; regenerate, see Section 4)
│   ├── Preprocessed_Synth2026/ Synth-2026 synthetic panel.
│   └── Raw_Data/
│       ├── README_sources.md   Provenance, DOIs, licenses, download links.
│       └── DescriptionDataCoSupplyChain.csv
│
├── results/
│   ├── aggregated_summary.json All per-seed results (9 models × 4 datasets × 10 seeds).
│   ├── sig_tests.json          Wilcoxon p-values, effect sizes, adjusted significance.
│   ├── budget_sensitivity.json A4 NR% at each budget scale (5 seeds × 5 scales).
│   └── *.csv                   Dataset and panel summary tables.
│
└── logs/
    ├── seeds_4751_out.txt      Console log of the seeds 47–51 training run.
    ├── sensitivity_out.txt     Console log of the budget-sensitivity run.
    └── watcher_out.txt         Log of the watcher that launched the sensitivity run.
```

---

## 2. Environment setup

**Requirements:** Python 3.11, NumPy, pandas, SciPy, scikit-learn, PyTorch,
XGBoost, CVXPY, MAPIE, matplotlib.

### Option A — conda (recommended)

```bash
conda env create -f environment.yml
conda activate sidn-cro
```

### Option B — pip

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

A CUDA-capable GPU is optional; all experiments also run on CPU (slower).

---

## 3. Reproducing the results

Run all commands **from the root of this package** (the folder containing
this README). The scripts add `src/` to the path automatically.

### 3a. Instant reproduction of the paper tables (no training)

The shipped `results/aggregated_summary.json` already contains every run.
To regenerate the Normalized-Regret, MAPE, and feasibility tables exactly as
reported in the paper, aggregate the saved results:

```bash
python code/run_all.py --aggregate_only
```

To reproduce the Wilcoxon significance table (Section "Results"):

```bash
python code/statistical_tests.py
```

### 3b. Full reproduction from scratch

Retrain and re-evaluate every configuration
(9 models × 4 datasets × 10 seeds = 360 runs):

```bash
python code/run_all.py
```

Defaults: `--datasets dataco olist hm synth`, `--models B0 B1 B2 B3 B4 B5 A1 A2 A4`,
`--seeds 42 43 44 45 46 47 48 49 50 51`, `--alpha 0.10` (90% conformal coverage),
`--rho 0.50` (CRO robustness level), `--sqp_method SLSQP`.

A single dataset/model/seed can be run for a quick check, e.g.:

```bash
python code/run_all.py --datasets dataco --models A4 --seeds 42
# or the lightweight single-seed runner:
python code/fast_run.py --datasets dataco --models B1 A1 A4 --seeds 42
```

### 3c. Budget-scale sensitivity analysis

```bash
python code/sensitivity_delta.py --datasets dataco olist synth
```

Produces `results/budget_sensitivity.json` (A4 NR% at budget scales
γ ∈ {0.50, 0.75, 1.00, 1.25, 1.50}, seeds 42–46).

---

## 4. Datasets

Three of the four datasets are shipped here in **preprocessed weekly-panel
form** (`data/Preprocessed*`), which is exactly what the models consume.

**H&M exception.** The H&M data are distributed under Kaggle competition rules,
which do not permit redistribution, so `data/Preprocessed_HM/` is not included.
To reproduce the H&M results, download the competition files from Kaggle into
`data/Raw_Data/` and run `src/data_pipeline/ingest_hm.py` followed by the
matching `make_train_val_test*.py` script. All other datasets reproduce without
any external download.

The three real-world datasets are public; full raw archives are **not**
redistributed here owing to their size (≈1 GB) and licensing terms. The raw
sources, DOIs, and licenses are documented in `data/Raw_Data/README_sources.md`:

| Dataset | Source | License |
|---|---|---|
| **DataCo Supply Chain** | Mendeley Data, DOI [10.17632/8gx2fvg2k6.3](https://doi.org/10.17632/8gx2fvg2k6.3) | CC BY 4.0 |
| **Olist Brazilian E-Commerce** | [Kaggle: olistbr/brazilian-ecommerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) | CC BY-NC-SA 4.0 |
| **H&M Personalized Fashion** | [Kaggle competition: h-and-m-personalized-fashion-recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations) | Competition rules |

To regenerate the preprocessed panels from raw downloads, place the raw files
under `data/Raw_Data/` (see `README_sources.md` for the expected file names) and
run the ingestion scripts in `src/data_pipeline/`
(`ingest_dataco.py`, `ingest_olist.py`, `ingest_hm.py`, `build_synth2026.py`,
followed by the `make_train_val_test*.py` and `synth_*_calibration*.py` scripts).

**Synth-2026** is purely synthetic and fully regenerable from
`src/data_pipeline/build_synth2026.py` (fixed seeds, no external data).

---

## 5. Hardware and runtime

The full 360-run reproduction (`run_all.py`) takes approximately **33–36 hours**
on a workstation with an NVIDIA RTX 5050 GPU and an Intel Core i7 CPU. The
instant aggregation (`--aggregate_only`) and the statistical tests complete in
seconds. Random seeds 42–51 are fixed for full determinism.

---

## 6. Mapping outputs → paper

| Paper element | Reproduced by |
|---|---|
| Normalized Regret (NR%) table | `run_all.py --aggregate_only` → `aggregated_summary.json` |
| MAPE% and Feasibility% tables | `run_all.py --aggregate_only` |
| Wilcoxon significance table | `statistical_tests.py` → `sig_tests.json` |
| Budget-scale sensitivity table | `sensitivity_delta.py` → `budget_sensitivity.json` |
| Ablation (B1→A1→A4) | per-model NR in `aggregated_summary.json` |
