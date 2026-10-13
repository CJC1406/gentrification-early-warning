# California Gentrification Early Warning

A data-driven dashboard for exploring historical housing-cost changes across California census tracts, comparing machine-learning models, and inspecting tract-level patterns using U.S. Census Bureau data.

> **Research limitation:** The current target labels a tract as positive when its median monthly housing cost increased between 2012 and 2016. This is a housing-cost-increase proxy, not a complete  forecast of future gentrification.

## Contents

- [Project overview](#project-overview)
- [Features](#features)
- [Data and target definition](#data-and-target-definition)
- [Technology stack](#technology-stack)
- [Repository structure](#repository-structure)
- [Requirements](#requirements)
- [Setup](#setup)
- [Data preparation and model training](#data-preparation-and-model-training)
- [Run the dashboard](#run-the-dashboard)
- [Outputs](#outputs)
- [Model evaluation](#model-evaluation)
- [Testing with new data](#testing-with-new-data)
- [Future improvements](#future-improvements)
- [Data sources](#data-sources)

## Project overview

Housing costs can change unevenly across neighborhoods. This project brings historical American Community Survey (ACS) estimates and census-tract geography together in an interactive dashboard. It helps users inspect where measured housing costs increased, compare tract values with California-wide statistics, and review machine-learning model results.

The current analysis compares the 2012 and 2016 ACS 5-year estimates. It uses earlier socioeconomic Census features to explore whether a model can distinguish tracts where the target housing-cost increase occurred. Model evaluation and geographic validation are included to help users understand how results vary under different validation strategies.

## Features

### Overview
- Counts of census tracts with valid matched housing-cost observations.
- Summary of tracts with rising housing costs.
- State-level and tract-level historical comparisons.
- Distribution of changes in monthly housing costs.
- Table of tracts with the largest observed increases.
- CSV download of tract-level data when available.

### Map
- Interactive California census-tract map using 2010 tract boundaries.
- Colouring by observed housing-cost change or available model output.
- Tract search by 11-digit Census GEOID.
- Clear distinction between tracts with rising costs, no increase, and missing matched data.

### Tract explorer
- Inspect a selected tract's 2012 and 2016 housing-cost values.
- Compare the selected tract with the California median.
- View the absolute and percentage change where valid.
- Show a model estimate only when a corresponding prediction is available.

### Model results
- Compare a majority-class baseline with logistic regression, random forest, neural network (ANN), and ensemble approaches.
- Review accuracy, balanced accuracy, precision, recall, F1, and ROC-AUC where calculated.
- Compare random cross-validation with county-grouped cross-validation for the random forest.
- Inspect saved model artefacts where available.

### Methodology
- Explain the housing-cost proxy target, feature preparation, validation approach, Census missing-value handling, and limitations.

## Data and target definition

The dashboard's historical target is derived from median monthly housing-cost estimates for each tract:

`housing_cost_change = housing_cost_2016 - housing_cost_2012`

The binary label is:

`gentrified = 1 if housing_cost_change > 0, otherwise 0`

In the implementation, the `gentrified` column is therefore a convenient legacy label name for a **positive housing-cost change**, not a verified gentrification label. Interpret results as “costs rose” or “no increase” rather than treating the label as evidence of gentrification.

The model-training scripts use `data/raw/model_dataset.csv`, which is built from DP03 socioeconomic features for 2010 and 2011 joined to the target. Those earlier features are intended to represent information available before the 2012–2016 target period. Verify that every feature and spatial feature is genuinely available at prediction time and does not encode target-period information before describing the output as an early warning.

The map uses 2010 Census tract boundaries. Tract identifiers are expected to be 11-digit GEOIDs and should be preserved as strings to avoid losing leading zeroes.

## Technology stack

- **Python** — data processing and model training
- **Streamlit** — interactive dashboard
- **Pandas / NumPy** — data preparation and numerical operations
- **GeoPandas / Shapely** — geographic data and tract polygons
- **Plotly** — interactive charts
- **Folium / streamlit-folium** — map display
- **scikit-learn** — preprocessing, classical models, validation, and metrics
- **TensorFlow / Keras** — artificial neural network
- **libpysal** — spatial analysis utilities
- **Requests** — Census API requests
- **Joblib** — model and preprocessing artefact persistence

## Repository structure

The following shows the main files. Local datasets, trained artefacts, and generated results may be ignored by Git and may not be included in a fresh clone.

```text
gentrification-early-warning/
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
├── .streamlit/
│   └── config.toml
├── src/
│   ├── download_census.py
│   ├── download_dp03.py
│   ├── build_dataset.py
│   ├── build_features.py
│   ├── build_target.py
│   ├── build_spatial_features.py
│   ├── clean_dataset.py
│   ├── preprocessing.py
│   ├── train_model.py
│   ├── train_logistic_model.py
│   ├── train_spatial_model.py
│   ├── train_ann.py
│   └── train_ensemble.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
└── results/
```

## Requirements

- Python **3.11** is the recommended starting point for a reproducible environment with the project's geospatial and TensorFlow dependencies. Compatibility can vary by macOS version and CPU architecture.
- Git
- Internet access to download public Census data or call the Census API.
- Sufficient disk space for Census data, geospatial files, and model artefacts.

The package versions in `requirements.txt` are minimum-version constraints, not a fully locked environment. For a research submission or deployment, test a clean installation and consider generating a lock file with exact versions after the pipeline has been verified.

## Setup

Run these commands from the repository root.

### 1. Clone the repository

```bash
git clone https://github.com/CJC1406/gentrification-early-warning.git
cd gentrification-early-warning
```

### 2. Create and activate a virtual environment

macOS / Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

If Python 3.11 is not installed, install it using your preferred Python distribution, then create the environment with that interpreter.

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

Check key imports:

```bash
python -c "import streamlit, pandas, numpy, sklearn, geopandas, plotly, tensorflow; print('Core imports OK')"
```

If installation fails, record your Python version, macOS version, CPU architecture, and the failing package before changing versions.

### 4. Configure Census API access if needed

Some download scripts require a Census API key. Set it as an environment variable in the terminal:

```bash
export CENSUS_API_KEY="YOUR_CENSUS_API_KEY"
```

Replace the placeholder with your own key. Do not commit API keys, `.env` files, or credentials to Git. Some public Census API requests work without a key, but scripts that call `census_api_key()` require the environment variable.

## Data preparation and model training

**Important:** A fresh clone may not contain the local data, shapefile, trained models, or generated result CSVs. The dashboard needs the files listed below. Do not assume that running `streamlit run app.py` alone creates them.

### Expected inputs

| File or path | Purpose |
|---|---|
| `data/raw/housing_cost_target.csv` | Tract-level 2012 and 2016 housing-cost values and binary target |
| `data/raw/dp03_2010_ca.csv` | 2010 ACS DP03 socioeconomic data |
| `data/raw/dp03_2011_ca.csv` | 2011 ACS DP03 socioeconomic data |
| `data/raw/acs5_2012_ca.csv` | ACS 2012 housing-cost variable used by target generation |
| `data/raw/acs5_2016_ca.csv` | ACS 2016 housing-cost variable used by target generation |
| `data/raw/tract10/tl_2010_06_tract10.shp` and companion shapefile files | 2010 California census-tract geometry |
| `data/raw/model_dataset.csv` | Combined training dataset |
| `data/processed/california_acs_2010_2016_clean.csv` | Cleaned historical ACS table used by the dashboard |

The actual downloader scripts and their expected filenames are the source of truth. Check their arguments and output paths before running them. Census shapefile ZIP files must be extracted so the `.shp`, `.shx`, `.dbf`, and `.prj` files are available together.

### Generate the target and training dataset

From the repository root, the target and feature-building scripts currently use paths relative to that directory. Once their raw input files are present, the relevant scripts include:

```bash
python src/build_target.py
python src/build_features.py
```

`build_target.py` creates `data/raw/housing_cost_target.csv`. `build_features.py` combines 2010 and 2011 DP03 data with that target to create `data/raw/model_dataset.csv`.

Other scripts in `src/` cover Census downloads, dataset assembly, cleaning, and spatial features. Inspect the script before running it and follow its actual input/output paths. In particular, `build_dataset.py` expects `data/raw/acs_2010.csv` through `data/raw/acs_2016.csv`; `clean_dataset.py` expects `data/processed/california_acs_2010_2016.csv`. These are separate inputs from the DP03 training dataset, so do not assume one command creates every required dashboard file.

### Train the models

The training scripts import `preprocessing` as a module from `src/`. Run them with `src` on the Python module path:

```bash
PYTHONPATH=src python src/train_logistic_model.py
PYTHONPATH=src python src/train_model.py
```

For the combined model comparison used by the dashboard:

```bash
PYTHONPATH=src python src/train_ann.py
PYTHONPATH=src python src/train_ensemble.py
```

The ANN training step may be computationally intensive. The ensemble script expects `data/raw/model_dataset.csv` and writes model artefacts to `models/` and evaluation files to `results/`.

For the spatial model, first ensure `data/raw/model_dataset_spatial.csv` exists and that the script's shapefile and spatial-feature inputs are available:

```bash
PYTHONPATH=src python src/train_spatial_model.py
```

Run the scripts only after their inputs have been generated. These commands describe the current scripts' expected usage; verify a clean end-to-end run before treating the workflow as fully reproducible.

## Run the dashboard

From the repository root, activate the environment and run:

```bash
streamlit run app.py
```

Streamlit will print a local URL, usually `http://localhost:8501`. Open it in a browser.

If the dashboard reports missing data, check the expected paths in `app.py`, prepare the missing inputs, and rerun the appropriate pipeline scripts. The app reads existing CSV and model artefacts; it is not itself the full data-ingestion and model-training pipeline.

## Outputs

The ensemble training script is designed to generate files including:

- `models/logistic_regression.joblib`
- `models/random_forest.joblib`
- `models/ann_preprocessor.joblib`
- `models/feature_columns.joblib`
- `models/ann_model.keras`
- `results/model_comparison.csv`
- `results/county_cv.csv`
- `results/tract_predictions.csv`

The dashboard also reads:

- `data/raw/housing_cost_target.csv`
- `data/processed/california_acs_2010_2016_clean.csv`
- `data/raw/tract10/tl_2010_06_tract10.shp`

Generated files may be excluded by `.gitignore`. If a user clones the repository, provide instructions for regenerating them or a stable download location with checksums.

## Model evaluation

The project compares model performance against a majority-class baseline. Metrics include:

- **Accuracy:** proportion of correct classifications.
- **Balanced accuracy:** average recall across classes, useful when class frequencies differ.
- **Precision:** fraction of positive predictions that are true positives.
- **Recall:** fraction of positive examples detected.
- **F1:** harmonic mean of precision and recall.
- **ROC-AUC:** ranking ability across classification thresholds when both classes are present.


<img width="1042" height="503" alt="Screenshot 2026-10-13 at 5 57 20 AM" src="https://github.com/user-attachments/assets/483a4888-fc10-4583-be57-594c8d34a708" />


<img width="1710" height="1112" alt="Screenshot 2026-10-13 at 5 56 59 AM" src="https://github.com/user-attachments/assets/052788ac-1ec1-40dd-83a7-6c016e8e3f62" />





The ensemble pipeline also compares ordinary stratified cross-validation with county-grouped cross-validation for a random forest. County-grouped evaluation is important because nearby tracts may be similar; random splitting alone can give an overly optimistic impression of performance on new places.

Do not compare metrics from different splits or pipelines as if they were directly equivalent. Record the dataset version, target definition, split, feature set, model settings, and package versions alongside results. Consider adding a precision-recall curve, confusion matrix, calibration check, and evaluation on a later time period or geographically held-out counties.

## Testing with new data

Use this workflow when you want to test the project with a newer ACS release, a different set of census tracts, or a separately prepared dataset. The exact commands depend on the inputs and paths expected by each script, so check the relevant script before running it.

### 1. Prepare a new dataset

- Download the relevant ACS estimates from the [U.S. Census Bureau](https://www.census.gov/programs-surveys/acs) or use the project's Census download scripts.
- Keep the required housing-cost variable and socioeconomic features, and document the ACS table, year, geography, and variable names used.
- Keep census-tract GEOIDs as **11-character strings**. Do not convert them to integers, because leading zeroes are meaningful.
- Check for duplicate GEOIDs, missing values, inconsistent column names, and changes in tract boundaries.
- Keep the new data separate from the original files until it has passed validation.

### 2. Build a consistent target

The current target uses the difference between 2012 and 2016 median monthly housing costs. For a new period, define the comparison explicitly, for example:

```text
housing_cost_change = housing_cost_later_period - housing_cost_earlier_period
positive_cost_change = 1 if housing_cost_change > 0, otherwise 0
```

Use comparable ACS estimates and the same unit and definition for both periods. This target indicates a rise in housing costs; it does **not** independently establish that gentrification or displacement occurred. If the new target period changes, record that change and do not compare its metrics directly with the original experiment without explaining the difference.

### 3. Rebuild features and the training dataset

Place new files in the paths expected by the scripts, or update the scripts to point to the new files. The existing workflow includes:

```bash
python src/build_target.py
python src/build_features.py
```

These commands expect the relevant raw inputs to be present. The current feature-building script is designed around the project's 2010 and 2011 DP03 files and the existing target, so it may need changes for newer periods or different input filenames. Do not assume these two commands automatically ingest arbitrary new data.

Before training, verify that:
- The target and feature rows match by GEOID.
- Each row represents one tract.
- The feature columns have the expected names and types.
- Missing-value handling is consistent with the training pipeline.
- Features do not use information from after the target's starting point or otherwise leak the outcome.
- Any geographic features use compatible tract boundaries.

### 4. Retrain the models

Once the dataset is validated, run the relevant training scripts from the repository root:

```bash
PYTHONPATH=src python src/train_logistic_model.py
PYTHONPATH=src python src/train_model.py
PYTHONPATH=src python src/train_ann.py
PYTHONPATH=src python src/train_ensemble.py
```

Run the spatial model only after its spatial dataset and geometry inputs are available:

```bash
PYTHONPATH=src python src/train_spatial_model.py
```

These scripts may overwrite existing files in `models/` and `results/`. Back up any results you want to preserve before retraining. Confirm that the output artefacts were generated successfully and that the dashboard loads the newly generated results.

### 5. Test on genuinely unseen data

For a meaningful test, hold out data that was not used to fit the model or tune its settings. For example, use a later time period or hold out entire counties. Do not use the same records for both training and final evaluation.

Check the results in `results/`, including `model_comparison.csv`, `county_cv.csv`, and `tract_predictions.csv` when those files are produced. Review the number of rows, missing predictions, class balance, and relevant metrics such as balanced accuracy, precision, recall, F1, and ROC-AUC where applicable. Compare the models against the majority-class baseline.

For a practical manual check:
1. Select a few GEOIDs from the held-out dataset.
2. Confirm their input features and target values against the source data.
3. Check that each GEOID has at most one prediction in the output.
4. Verify that predicted values and probabilities, if provided, are valid.
5. Compare the predictions with the observed target only after predictions have been generated.
6. Record the data period, held-out geography, features, model, and metrics.

A successful dashboard launch only confirms that the app can run; it does not prove that the model generalizes to new data. The current scripts and file paths may require adjustment before a fully automated new-data workflow is available.

## Future improvements

Possible next steps for this project include:

- **Support newer ACS periods:** parameterize the years and variable names so the data pipeline can be reused without manually editing several scripts.
- **Automate data validation:** check GEOID format, duplicate tracts, missing values, expected columns, and row counts before training.
- **Add a reproducible pipeline command:** provide one documented entry point to download or load data, validate it, build features, train models, and produce evaluation reports.
- **Strengthen evaluation:** use county-held-out and time-based tests, and report confusion matrices, precision-recall curves, and probability calibration where appropriate.
- **Track experiments:** save dataset periods, target definitions, feature lists, package versions, model settings, and metrics with each run.
- **Improve interpretability:** show which features influence model outputs, while explaining that feature importance does not establish causation.
- **Account for inflation:** provide an inflation-adjusted housing-cost comparison in addition to nominal dollar changes.
- **Handle geographic changes:** document how census-tract boundary changes are matched or harmonized across periods.
- **Improve the target definition:** consider multiple indicators of neighborhood change, affordability, and displacement where reliable data is available. Keep proxy labels clearly distinguished from verified outcomes.
- **Add automated tests:** test data cleaning, GEOID matching, target construction, feature alignment, and prediction output generation on small known examples.

Prioritize **data validation, leakage-resistant evaluation, and a repeatable new-data pipeline** before describing the dashboard as a reliable early-warning system.

## Data sources

- **U.S. Census Bureau — American Community Survey (ACS):** socioeconomic and housing estimates. https://www.census.gov/programs-surveys/acs
- **U.S. Census Bureau — Census API:** https://www.census.gov/data/developers/data-sets.html
- **U.S. Census Bureau — TIGER/Line shapefiles:** census geography boundaries. https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html

ACS 5-year estimates describe multi-year periods; they should not be interpreted as single-year measurements. Housing-cost dollar values in the historical dashboard are nominal unless an inflation adjustment is explicitly applied.




Vivaan J
Chiranth J
