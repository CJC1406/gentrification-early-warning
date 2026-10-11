"""
Final comparison: Logistic Regression, Random Forest, ANN and ensembles.

Outputs
  models/logistic_regression.joblib   imputer + scaler + model pipeline
  models/random_forest.joblib         imputer + model pipeline
  models/ann_preprocessor.joblib      imputer + scaler used by the ANN
  models/ann_model.keras
  models/feature_columns.joblib
  results/model_comparison.csv        held-out test metrics (read by app.py)
  results/county_cv.csv               county-grouped CV check (read by app.py)
  results/tract_predictions.csv       out-of-fold RF probability per tract (read by app.py)
"""

import json
import random

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedGroupKFold,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from preprocessing import (
    DATA_DIR,
    MODEL_DIR,
    RESULTS_DIR,
    SEED,
    county_groups,
    evaluate,
    load_xy,
    majority_baseline,
    make_preprocessor,
    print_results,
    split,
)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_ANN = {"layers": [256, 128, 64], "dropout": 0.20, "learning_rate": 0.001, "batch_size": 32}


def load_ann_config():
    path = RESULTS_DIR / "ann_best_config.json"
    if path.exists():
        config = json.loads(path.read_text())
        print(f"Using ANN config from {path}: {config.get('name')}")
        return config
    print("results/ann_best_config.json not found; run train_ann.py first. Using default ANN config.")
    return DEFAULT_ANN


def make_rf():
    return Pipeline(
        [
            ("prep", make_preprocessor(scale=False)),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=125,
                    max_features=18,
                    random_state=SEED,
                    n_jobs=-1,
                ),
            ),
        ]
    )


# ============================================================
# DATA
# ============================================================

X, y, tract_ids = load_xy(DATA_DIR / "model_dataset.csv")
X_train, X_test, y_train, y_test, ids_train, ids_test = split(X, y, tract_ids)

print(f"Training samples: {len(X_train):,}")
print(f"Test samples:     {len(X_test):,}")
print(f"Features:         {X.shape[1]:,}")
print(f"Positive rate:    {y.mean():.3f}")

# ============================================================
# LOGISTIC REGRESSION (C tuned by CV on the training set)
# ============================================================

lr_search = GridSearchCV(
    Pipeline(
        [
            ("prep", make_preprocessor(scale=True)),
            ("lr", LogisticRegression(max_iter=2000, random_state=SEED)),
        ]
    ),
    param_grid={"lr__C": [0.001, 0.005, 0.01, 0.05, 0.1, 1.0]},
    scoring="roc_auc",
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
    n_jobs=-1,
)
lr_search.fit(X_train, y_train)
logistic = lr_search.best_estimator_
print(f"Logistic regression C = {lr_search.best_params_['lr__C']}")

lr_prob = logistic.predict_proba(X_test)[:, 1]
lr_pred = (lr_prob >= 0.5).astype(int)

# ============================================================
# RANDOM FOREST
# ============================================================

random_forest = make_rf().fit(X_train, y_train)
rf_prob = random_forest.predict_proba(X_test)[:, 1]
rf_pred = (rf_prob >= 0.5).astype(int)

# ============================================================
# ANN
# ============================================================

ann_config = load_ann_config()
ann_prep = make_preprocessor(scale=True).fit(X_train)
X_train_s = ann_prep.transform(X_train)
X_test_s = ann_prep.transform(X_test)

X_tr, X_es, y_tr, y_es = train_test_split(
    X_train_s, y_train, test_size=0.20, random_state=SEED, stratify=y_train
)

ann = Sequential([Input(shape=(X_train_s.shape[1],))])
for units in ann_config["layers"]:
    ann.add(Dense(units, activation="relu"))
    ann.add(Dropout(ann_config["dropout"]))
ann.add(Dense(1, activation="sigmoid"))
ann.compile(
    optimizer=Adam(learning_rate=ann_config["learning_rate"]),
    loss="binary_crossentropy",
    metrics=["AUC"],
)
ann.fit(
    X_tr,
    y_tr,
    validation_data=(X_es, y_es),
    epochs=100,
    batch_size=ann_config["batch_size"],
    callbacks=[EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)],
    verbose=0,
)

ann_prob = ann.predict(X_test_s, verbose=0).ravel()
ann_pred = (ann_prob >= 0.5).astype(int)

# ============================================================
# ENSEMBLES
# ============================================================

vote_pred = (np.column_stack([lr_pred, rf_pred, ann_pred]).sum(axis=1) >= 2).astype(int)
vote_score = np.column_stack([lr_pred, rf_pred, ann_pred]).mean(axis=1)

soft_prob = np.column_stack([lr_prob, rf_prob, ann_prob]).mean(axis=1)
soft_pred = (soft_prob >= 0.5).astype(int)

# ============================================================
# RESULTS
# ============================================================

baseline_row, _, _ = majority_baseline(y_train, y_test)

rows = [
    baseline_row,
    evaluate("Logistic Regression", y_test, lr_pred, lr_prob),
    evaluate("Random Forest", y_test, rf_pred, rf_prob),
    evaluate("ANN", y_test, ann_pred, ann_prob),
    evaluate("Ensemble (majority vote)", y_test, vote_pred, vote_score),
    evaluate("Ensemble (average probability)", y_test, soft_pred, soft_prob),
]

print()
print("=" * 65)
print("MODEL COMPARISON (held-out test set)")
print("=" * 65)
results_df = print_results(rows)
results_df.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)

tn, fp, fn, tp = confusion_matrix(y_test, vote_pred).ravel()
print(f"\nMajority-vote confusion matrix: TN={tn} FP={fp} FN={fn} TP={tp}")

# ============================================================
# ROBUSTNESS: COUNTY-GROUPED CROSS-VALIDATION
#
# Neighbouring tracts are similar, so a random split puts near-
# duplicates on both sides. Holding out whole counties gives a more
# honest estimate of performance on unseen places.
# ============================================================

groups = county_groups(tract_ids)
cv_rows = []
for cv_name, cv in [
    ("Random 5-fold", StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)),
    ("County-grouped 5-fold", StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)),
]:
    scores = cross_val_score(
        make_rf(),
        X,
        y,
        groups=groups if isinstance(cv, StratifiedGroupKFold) else None,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
    )
    cv_rows.append(
        {"Model": "Random Forest", "Validation": cv_name, "ROC AUC mean": scores.mean(), "ROC AUC std": scores.std()}
    )

print("\nRandom-split vs county-grouped cross-validation")
cv_df = print_results(cv_rows)
cv_df.to_csv(RESULTS_DIR / "county_cv.csv", index=False)

# ============================================================
# PER-TRACT OUT-OF-FOLD PREDICTIONS (for the dashboard)
#
# Each tract's probability comes from a model that never saw it.
# ============================================================

oof_prob = cross_val_predict(
    make_rf(),
    X,
    y,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
    method="predict_proba",
    n_jobs=-1,
)[:, 1]

pd.DataFrame(
    {
        "tract_id": tract_ids.to_numpy(),
        "rf_oof_probability": oof_prob,
        "actual_label": y.to_numpy(),
        "in_test_set": tract_ids.isin(ids_test).to_numpy(),
    }
).to_csv(RESULTS_DIR / "tract_predictions.csv", index=False)

# ============================================================
# SAVE MODELS
# ============================================================

joblib.dump(logistic, MODEL_DIR / "logistic_regression.joblib")
joblib.dump(random_forest, MODEL_DIR / "random_forest.joblib")
joblib.dump(ann_prep, MODEL_DIR / "ann_preprocessor.joblib")
joblib.dump(X.columns.tolist(), MODEL_DIR / "feature_columns.joblib")
ann.save(MODEL_DIR / "ann_model.keras")

print("\nSaved models to:", MODEL_DIR)
print("Saved results to:", RESULTS_DIR)
