

# California Gentrification Early Warning System

A machine learning and geospatial analysis project that explores how housing costs and neighborhood characteristics can help identify areas vulnerable to gentrification across California.

## Overview

Gentrification can increase housing costs and place pressure on existing residents. Identifying neighborhoods experiencing these changes can help researchers and policymakers investigate potential housing affordability concerns.

This project uses historical U.S. Census and American Community Survey (ACS) data to analyze California census tracts, compare machine learning models, and visualize historical housing-cost patterns through an interactive map.

The project explores whether socioeconomic indicators and spatial relationships between neighboring census tracts can improve the identification of areas experiencing rising housing costs.

## Objectives

- Analyze historical housing and socioeconomic data across California.
- Develop machine learning models to classify census tracts based on housing-cost changes.
- Compare traditional machine learning, neural networks, and ensemble methods.
- Engineer spatial features to examine relationships between neighboring tracts.
- Visualize geographic patterns through an interactive California map.
- Explore the potential of data-driven approaches for housing affordability research.

## Machine Learning Models

The project implements the following approaches:

| Model | Purpose |
|---|---|
| Logistic Regression | Establishes a linear classification baseline. |
| Random Forest | Captures nonlinear relationships among census indicators. |
| Spatial Random Forest | Adds neighborhood-level spatial features. |
| Artificial Neural Network (ANN) | Learns nonlinear patterns using multiple neural network layers. |
| Majority-Vote Ensemble | Combines predictions from Logistic Regression, Random Forest, and ANN. |

Models are evaluated using accuracy, precision, and recall.

## Spatial Analysis

Four engineered spatial features are used to represent relationships between adjacent census tracts:

- **Housing spatial lag:** Average historical housing-cost change among neighboring tracts.
- **Income spatial lag:** Average change in neighboring income-distribution indicators.
- **Housing Local Moran's I:** Measures local spatial association in housing-cost changes.
- **Income Local Moran's I:** Measures local spatial association in income-distribution changes.

Census tracts are connected using Queen contiguity, which considers tracts that share an edge or a corner.

## Data

The project uses historical census and housing data for California, including:

- Housing-cost indicators.
- Demographic and socioeconomic characteristics.
- Income-distribution indicators.
- Census tract geographic boundaries.
- Engineered spatial features.

The current historical target uses the change in a selected housing-cost measure between 2012 and 2016:

`housing_cost_change = housing_cost_2016 - housing_cost_2012`

A tract is assigned a positive label when this change is greater than zero.

**Important:** Rising housing costs are used as a simplified proxy for gentrification. They do not independently establish displacement, neighborhood demographic transformation, or the causes of housing-cost increases.

## Interactive Dashboard

The Streamlit application provides a geographic interface for exploring the research.

Features include:

- Interactive California census-tract map.
- Historical housing-cost comparisons.
- Census tract selection and lookup.
- Neighborhood-level socioeconomic indicators.
- Model evaluation and methodology sections.

The dashboard is a research prototype. A map label based on historical housing costs should not be interpreted as a validated prediction of future gentrification.

## Technology Stack

- Python
- Pandas and NumPy
- Scikit-learn
- TensorFlow / Keras
- GeoPandas
- PySAL / libpysal
- Streamlit
- Folium
- Plotly
- U.S. Census Bureau API

## Project Structure

```text
gentrification-early-warning/
├── app.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── results/
├── src/
│   ├── train_model.py
│   ├── train_logistic_model.py
│   ├── train_spatial_model.py
│   ├── train_ann.py
│   ├── train_ensemble.py
│   └── build_spatial_features.py
├── requirements.txt
└── README.md
```

The exact structure may differ depending on the current repository. Large datasets and trained model artifacts may be stored locally rather than committed to GitHub.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/gentrification-early-warning.git
cd gentrification-early-warning
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare the data

Place the required census datasets, target CSV, and geographic files in their expected directories. Consult the training scripts for the required filenames and columns.

### 5. Run the application

```bash
streamlit run app.py
```

## Model Training

Run the relevant scripts from the repository root after preparing the datasets:

```bash
python src/train_model.py
python src/train_logistic_model.py
python src/train_spatial_model.py
python src/train_ann.py
python src/train_ensemble.py
```

The scripts require their corresponding input datasets. Check the paths and filenames before running them.

## Evaluation

The models are compared using:

- **Accuracy:** Proportion of correct classifications.
- **Precision:** Proportion of predicted positive cases that are correct.
- **Recall:** Proportion of actual positive cases correctly identified.

A reliable comparison requires consistent target definitions, test data, and feature preparation. The current project has reported approximately 65–67% accuracy for some model experiments; individual results should be reported with the corresponding experiment and dataset.

## Limitations

- Housing-cost increases alone do not fully define gentrification.
- Historical census estimates may contain uncertainty.
- Spatial relationships do not establish causation.
- Model results depend on feature selection, data quality, and evaluation design.
- Features must be restricted to information available before the prediction period to support a genuine early-warning claim.
- Spatial and geographic data should be checked for leakage between training and test sets.

## Future Improvements

- Evaluate models using strictly time-separated training and test periods.
- Add more direct indicators of displacement and housing affordability.
- Improve spatial weighting using distances between census tracts.
- Calibrate predicted probabilities and analyze false positives and false negatives.
- Integrate validated future-period predictions into the interactive map.
- Improve model interpretability for policy and research applications.

## References

This project is based  on:

**A Proximity-Based Early Warning System for Gentrification in California** (2018), Aakash Pattabi, Stanford University.

The original research explores census-based classification, Random Forests, XGBoost, LASSO logistic regression, and spatial features to investigate the early indicators of gentrification.

Vivaan J 
Chiranth J
