# California Gentrification Early Warning

A data-driven dashboard for exploring historical housing-cost changes across California census tracts, comparing machine-learning models, and inspecting tract-level patterns using U.S. Census Bureau data.

> **Research limitation:** The current target labels a tract as positive when its median monthly housing cost increased between 2012 and 2016. This is a housing-cost-increase proxy, not a complete measure of gentrification, displacement, demographic change, or housing unaffordability. The dashboard is a historical research tool; its model scores should not be interpreted as proof that displacement occurred or as a validated forecast of future gentrification.

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
- [Data sources](#data-sources)
- [Limitations and responsible use](#limitations-and-responsible-use)
- [Troubleshooting](#troubleshooting)
- [Future improvements](#future-improvements)

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

The ensemble pipeline also compares ordinary stratified cross-validation with county-grouped cross-validation for a random forest. County-grouped evaluation is important because nearby tracts may be similar; random splitting alone can give an overly optimistic impression of performance on new places.

Do not compare metrics from different splits or pipelines as if they were directly equivalent. Record the dataset version, target definition, split, feature set, model settings, and package versions alongside results. Consider adding a precision-recall curve, confusion matrix, calibration check, and evaluation on a later time period or geographically held-out counties.

## Data sources

- **U.S. Census Bureau — American Community Survey (ACS):** socioeconomic and housing estimates. https://www.census.gov/programs-surveys/acs
- **U.S. Census Bureau — Census API:** https://www.census.gov/data/developers/data-sets.html
- **U.S. Census Bureau — TIGER/Line shapefiles:** census geography boundaries. https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html

ACS 5-year estimates describe multi-year periods; they should not be interpreted as single-year measurements. Housing-cost dollar values in the historical dashboard are nominal unless an inflation adjustment is explicitly applied.

## Limitations and responsible use

1. **Proxy target:** an increase in median housing costs is not, by itself, a complete definition of gentrification or displacement.
2. **Historical period:** the dashboard compares historical estimates; it is not a guarantee of future conditions.
3. **Nominal dollars:** displayed changes are not inflation-adjusted unless stated otherwise.
4. **ACS uncertainty:** ACS estimates have sampling error and may include missing or suppressed values.
5. **Geographic consistency:** tract boundaries and identifiers can change across Census vintages; the current map uses 2010 tract geography.
6. **Model uncertainty:** scores depend on feature selection, preprocessing, class balance, validation design, and model calibration.
7. **Equity and policy:** do not use tract-level model outputs alone to make decisions affecting individuals, housing access, enforcement, or allocation of essential services.
8. **Reproducibility:** results are only reproducible when the underlying data, model artefacts, code version, and environment are available.

## Troubleshooting

### `ModuleNotFoundError: preprocessing`

Run training scripts from the repository root with `PYTHONPATH=src`, for example:

```bash
PYTHONPATH=src python src/train_ensemble.py
```

### Missing CSV or shapefile

Confirm the path and filename in `app.py` or the relevant `src/` script. Extract the TIGER/Line ZIP into the expected `data/raw/tract10/` directory and preserve all companion shapefile files.

### `CENSUS_API_KEY` is missing

Set the environment variable in the same terminal session before running a script that requires it:

```bash
export CENSUS_API_KEY="YOUR_CENSUS_API_KEY"
```

### TensorFlow or GeoPandas installation errors

Use a supported Python version in a fresh virtual environment. Avoid mixing system packages with the project environment. Review the error and test compatible dependency versions instead of blindly upgrading every package.

### Dashboard shows no model estimate

Check whether `results/tract_predictions.csv` exists, includes `tract_id` and `rf_oof_probability`, and contains the selected tract's GEOID. Train the ensemble if the output is absent.

## Future improvements

- Establish a formal, literature-grounded gentrification or displacement target rather than relying only on housing-cost increases.
- Add a genuinely prospective evaluation: train on earlier years and test on a later period.
- Audit spatial leakage and evaluate on held-out counties or regions.
- Add model calibration, uncertainty intervals, feature explanations, and subgroup error analysis.
- Provide a tested one-command pipeline for downloading, validating, preparing, training, and launching.
- Add automated tests and continuous integration.
- Publish data provenance, licensing, and a versioned release of any distributable data/model artefacts.

## License

No license is specified unless a `LICENSE` file is added to this repository. Until then, do not assume that the source code is licensed for unrestricted reuse.
