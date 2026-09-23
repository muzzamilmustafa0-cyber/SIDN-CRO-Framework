# Raw Data Sources

## Primary dataset

**DataCo SMART SUPPLY CHAIN FOR BIG DATA ANALYSIS** (v3, March 2019)

| Field        | Value |
|--------------|-------|
| File         | `DataCoSupplyChainDataset.csv` |
| Description  | `DescriptionDataCoSupplyChain.csv` |
| Repository   | Mendeley Data |
| DOI          | [10.17632/8gx2fvg2k6.3](https://doi.org/10.17632/8gx2fvg2k6.3) |
| License      | CC BY 4.0 |
| Size         | 91.5 MB / 180 519 rows / 53 fields |
| Time span    | 2015-01-01 to 2018-01-31 |
| Citation     | Constante, F.; Silva, F.; Pereira, A. *DataCo SMART SUPPLY CHAIN FOR BIG DATA ANALYSIS*. Mendeley Data, V3, 2019. |

## Download

The raw CSV is downloaded automatically (via `curl`) from
`https://data.mendeley.com/public-files/datasets/8gx2fvg2k6/files/<id>/file_downloaded`
using the file IDs listed in the Mendeley REST endpoint
`/api/datasets/8gx2fvg2k6/files?version=3`.

## Calibrated overlays

The DataCo CSV does not contain advertising spend or circular-input fractions.
These are generated as **calibrated synthetic overlays** by the scripts:

* `src/data_pipeline/synth_ad_calibration.py`
   — cooperative ad spend $b_s, b_m, b_r$ per (retailer, week)
   — calibrated to the apparel ad-to-sales ratio of 4 - 12 % per period.

* `src/data_pipeline/synth_circular_calibration.py`
   — circular fractions $u_s, u_m, u_r$ per (retailer, week)
   — calibrated to Eurostat textile circular-material-use rates (2015 - 2023).

The resulting calibrated benchmark is released in
`data/Preprocessed/Train|Validation|Test/` and is fully reproducible from the
random seeds 42 / 43 / 44 fixed in the three calibration scripts.
