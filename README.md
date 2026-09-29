# Machine-learning analyses for blood-based lymphoma detection

Analysis code accompanying the manuscript. This repository contains the statistical workflow only; no patient-level study data are included.

The script validates a two-class Excel dataset, fits a standardized logistic-regression model, performs nested stratified cross-validation, selects each outer-fold threshold using only inner out-of-fold predictions from the outer training set, and reports pooled outer out-of-fold ROC AUC.

The pooled OOF AUC uses a DeLong normal-approximation 95% confidence interval. Sensitivity, specificity, and accuracy use stratified percentile bootstrap intervals based on the fixed OOF predictions. These are approximate internal-validation intervals and do not fully account for model refitting or overlap among cross-validation training sets.

## Requirements

The workflow was tested with Python 3.12 and the package versions pinned in `requirements.txt`.

```bash
python -m pip install -r requirements.txt
```

## Input

The repository includes `data/example_input.xlsx`, a synthetic demonstration dataset containing no patient records. The files under `results/example_run/` were generated exclusively from this synthetic dataset and are included only to show the expected output structure.

For each analysis, the input Excel file should contain at least two columns:

1. Column A: binary outcome (`0/1`; `Healthy/MALT+` is also accepted for the first analysis below)
2. Column B: continuous numeric predictor used for the corresponding analysis

The predictor in Column B differs across the manuscript analyses (e.g., MALPPip fluorescence intensity, β2-microglobulin, ferritin, or lactate dehydrogenase [LDH]). The same analysis pipeline is applied separately to each predefined classification task and predictor.

Each row must represent one independent sample.

The manuscript analyses use the following class definitions:

| Analysis | Class 0 | Class 1 | Predictor |
| --- | --- | --- | --- |
| MALT1-active lymphoma versus healthy controls | Healthy | MALT1-active lymphoma | MALPPip fluorescence intensity |
| MALT1-active versus MALT1-low/negative lymphoma | MALT1-low/negative lymphoma | MALT1-active lymphoma | MALPPip fluorescence intensity |
| CR versus non-CR using MALPPip | CR | non-CR | MALPPip fluorescence intensity |
| CR versus non-CR using beta-2-microglobulin | CR | non-CR | beta-2-microglobulin |
| CR versus non-CR using ferritin | CR | non-CR | ferritin |
| CR versus non-CR using LDH | CR | non-CR | lactate dehydrogenase |

For a demonstration run, keep the default settings at the top of `Machine_learning_analyses.py`:

```python
PATH_FILE = "data/example_input.xlsx"
OUT_DIR = "results"
```

For study data, change `PATH_FILE` to the local workbook location. Do not commit real patient-level or otherwise identifiable data to this repository.

## Run

```bash
python Machine_learning_analyses.py
```

Results are written to `results/run_YYYYMMDD_HHMMSS_microseconds/`.

Generated `results/run_*` directories are ignored by Git. `results/example_run/` is the only committed output directory and contains synthetic demonstration results.

## Analysis assumptions and outputs

- Outer validation uses stratified 5-fold cross-validation.
- The classification threshold for each outer fold is selected only from inner OOF predictions generated within that outer training set.
- The primary AUC is the pooled outer OOF AUC, not the mean of fold-specific AUCs.
- The AUC 95% CI uses the DeLong normal approximation on pooled OOF scores.
- Sensitivity, specificity, and accuracy use stratified percentile bootstrap intervals from fixed OOF predictions.
- These intervals do not fully account for repeated refitting, re-splitting, or threshold-reselection uncertainty.
- The script assumes independent samples and is not intended for repeated measurements from the same patient.

The repository intentionally excludes raw study data, patient identifiers, and manuscript-specific private file paths. Only outputs generated from the synthetic demonstration dataset are included.

## Data confidentiality

Do not commit real patient-level data to this repository. Study data must remain in an approved secure environment and are subject to the access restrictions described in the manuscript's Data Availability statement.

## Intended use

This code is provided for research reproducibility and is not a clinical decision-support tool.
