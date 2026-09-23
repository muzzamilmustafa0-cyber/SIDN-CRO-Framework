"""Centralised path helpers so every script writes to the right place."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "Raw_Data"
PREPROCESSED_DIR = PROJECT_ROOT / "data" / "Preprocessed"
TRAIN_DIR = PREPROCESSED_DIR / "Train"
VAL_DIR = PREPROCESSED_DIR / "Validation"
TEST_DIR = PREPROCESSED_DIR / "Test"
INTERMEDIATE_DIR = PREPROCESSED_DIR / "_intermediate"

RESULTS_DIR = PROJECT_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"

LOGS_DIR = PROJECT_ROOT / "logs"

for d in (TRAIN_DIR, VAL_DIR, TEST_DIR, INTERMEDIATE_DIR, TABLES_DIR, FIGURES_DIR, LOGS_DIR):
    d.mkdir(parents=True, exist_ok=True)

DATACO_CSV = RAW_DIR / "DataCoSupplyChainDataset.csv"

# --- H&M Personalised Fashion paths ----------------------------------------
HM_DIR              = RAW_DIR / "hm"
HM_ARTICLES         = HM_DIR / "articles.parquet"
HM_CUSTOMERS        = HM_DIR / "customers.parquet"
HM_TRANSACTIONS     = HM_DIR / "transactions_train.parquet"

HM_PREPROCESSED_DIR = PROJECT_ROOT / "data" / "Preprocessed_HM"
HM_TRAIN_DIR        = HM_PREPROCESSED_DIR / "Train"
HM_VAL_DIR          = HM_PREPROCESSED_DIR / "Validation"
HM_TEST_DIR         = HM_PREPROCESSED_DIR / "Test"
HM_INTERMEDIATE_DIR = HM_PREPROCESSED_DIR / "_intermediate"
for d in (HM_TRAIN_DIR, HM_VAL_DIR, HM_TEST_DIR, HM_INTERMEDIATE_DIR):
    d.mkdir(parents=True, exist_ok=True)

# --- Synthetic 2024-2026 stress-test panel paths ---------------------------
SYNTH26_PREPROCESSED_DIR = PROJECT_ROOT / "data" / "Preprocessed_Synth2026"
SYNTH26_TRAIN_DIR        = SYNTH26_PREPROCESSED_DIR / "Train"
SYNTH26_VAL_DIR          = SYNTH26_PREPROCESSED_DIR / "Validation"
SYNTH26_TEST_DIR         = SYNTH26_PREPROCESSED_DIR / "Test"
SYNTH26_INTERMEDIATE_DIR = SYNTH26_PREPROCESSED_DIR / "_intermediate"
for d in (SYNTH26_TRAIN_DIR, SYNTH26_VAL_DIR, SYNTH26_TEST_DIR, SYNTH26_INTERMEDIATE_DIR):
    d.mkdir(parents=True, exist_ok=True)

# --- Olist cross-replication paths -----------------------------------------
OLIST_DIR              = RAW_DIR / "olist"
OLIST_ORDERS_CSV       = OLIST_DIR / "olist_orders_dataset.csv"
OLIST_ITEMS_CSV        = OLIST_DIR / "olist_order_items_dataset.csv"
OLIST_PRODUCTS_CSV     = OLIST_DIR / "olist_products_dataset.csv"
OLIST_CUSTOMERS_CSV    = OLIST_DIR / "olist_customers_dataset.csv"
OLIST_TRANSLATION_CSV  = OLIST_DIR / "product_category_name_translation.csv"

# Olist preprocessed splits live in a sibling subtree so the two datasets
# never overwrite each other.
OLIST_PREPROCESSED_DIR = PROJECT_ROOT / "data" / "Preprocessed_Olist"
OLIST_TRAIN_DIR        = OLIST_PREPROCESSED_DIR / "Train"
OLIST_VAL_DIR          = OLIST_PREPROCESSED_DIR / "Validation"
OLIST_TEST_DIR         = OLIST_PREPROCESSED_DIR / "Test"
OLIST_INTERMEDIATE_DIR = OLIST_PREPROCESSED_DIR / "_intermediate"

for d in (OLIST_TRAIN_DIR, OLIST_VAL_DIR, OLIST_TEST_DIR, OLIST_INTERMEDIATE_DIR):
    d.mkdir(parents=True, exist_ok=True)
