# California Gentrification Early Warning

Predicts which California census tracts saw median monthly housing costs
(ACS table B25105) rise between 2012 and 2016, using only 2010–2011
ACS features, and presents the results in a Streamlit dashboard.

## Setup

```bash
pip install -r requirements.txt
export CENSUS_API_KEY=your_key      # https://api.census.gov/data/key_signup.html
```

## Run order (from the repository root)

```bash
python src/download_census.py          # B25105 housing cost, 2010/2011/2012/2016
python src/download_dp03.py            # DP03 economic profile, same years
python src/build_target.py             # data/raw/housing_cost_target.csv
python src/build_features.py           # data/raw/model_dataset.csv

# Spatial features need the 2010 TIGER/Line tract shapefile at
# data/raw/tl_2010_06_tract10.zip
python src/build_spatial_features.py   # data/raw/model_dataset_spatial.csv

python src/train_model.py              # random forest
python src/train_logistic_model.py     # logistic regression (C tuned by CV)
python src/train_spatial_model.py      # RF with vs. without spatial features
python src/train_ann.py                # ANN tuning on a validation split
python src/train_ensemble.py           # final comparison -> results/, models/

streamlit run app.py
```

All training scripts share `src/preprocessing.py`: the same Census
sentinel cleaning, the same 80/20 stratified split, train-only
imputation/scaling, and the same metrics (including ROC AUC and a
majority-class baseline).

`build_dataset.py`, `clean_dataset.py`, `create_target.py` and
`check_data.py` are earlier exploration scripts. They expect
`data/raw/acs_<year>.csv` files that no script in this repository
creates. The dashboard only uses their output for the optional
"Census indicators" table.
