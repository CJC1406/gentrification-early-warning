import random
import joblib
from pathlib import Path
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd
import tensorflow as tf

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# DATA
# ============================================================

DATA_DIR = Path("data/raw")
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(
    DATA_DIR / "model_dataset.csv"
)

y = df["gentrified"]


# ============================================================
# FEATURES
# ============================================================

X = df.drop(
    columns=[
        "tract_id",
        "housing_cost_change",
        "gentrified",
    ]
)

# Convert to numeric
X = X.apply(
    pd.to_numeric,
    errors="coerce"
)

# Replace Census missing-value sentinel
X = X.replace(
    999999999,
    np.nan
)

# Median imputation
X = X.fillna(
    X.median()
)

# Final safety check
X = X.fillna(0)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=SEED,
    stratify=y,
)


print(
    f"Training samples: {len(X_train):,}"
)

print(
    f"Test samples:     {len(X_test):,}"
)

print(
    f"Features:         {X.shape[1]:,}"
)


# ============================================================
# STANDARDIZATION
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)


# ============================================================
# LOGISTIC REGRESSION
# ============================================================

logistic = LogisticRegression(
    C=0.005,
    max_iter=2000,
    random_state=SEED,
)

logistic.fit(
    X_train_scaled,
    y_train
)

logistic_pred = logistic.predict(
    X_test_scaled
)


# ============================================================
# RANDOM FOREST
# ============================================================

random_forest = RandomForestClassifier(
    n_estimators=125,
    max_features=18,
    random_state=SEED,
    n_jobs=-1,
)

random_forest.fit(
    X_train,
    y_train
)

rf_pred = random_forest.predict(
    X_test
)


# ============================================================
# ANN
# BEST CONFIGURATION FROM YOUR TUNING
#
# 256 -> 128 -> 64
# Dropout = 0.20
# Learning rate = 0.001
# Batch size = 32
# ============================================================

ann = Sequential(
    [
        Dense(
            256,
            activation="relu",
            input_shape=(X_train_scaled.shape[1],),
        ),

        Dropout(0.20),

        Dense(
            128,
            activation="relu",
        ),

        Dropout(0.20),

        Dense(
            64,
            activation="relu",
        ),

        Dropout(0.20),

        Dense(
            1,
            activation="sigmoid",
        ),
    ]
)


ann.compile(
    optimizer=Adam(
        learning_rate=0.001
    ),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)


early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True,
)


ann.fit(
    X_train_scaled,
    y_train,
    validation_split=0.20,
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=0,
)


# Convert probabilities to binary predictions
ann_prob = ann.predict(
    X_test_scaled,
    verbose=0
).ravel()

ann_pred = (
    ann_prob >= 0.5
).astype(int)


# ============================================================
# MAJORITY-VOTE ENSEMBLE
# ============================================================

prediction_matrix = np.column_stack(
    [
        logistic_pred,
        rf_pred,
        ann_pred,
    ]
)

ensemble_pred = (
    prediction_matrix.sum(axis=1) >= 2
).astype(int)


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    name,
    y_true,
    predictions,
):

    accuracy = accuracy_score(
        y_true,
        predictions,
    )

    precision = precision_score(
        y_true,
        predictions,
    )

    recall = recall_score(
        y_true,
        predictions,
    )

    return {
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
    }


# ============================================================
# RESULTS
# ============================================================

results = [

    evaluate_model(
        "Logistic Regression",
        y_test,
        logistic_pred,
    ),

    evaluate_model(
        "Random Forest",
        y_test,
        rf_pred,
    ),

    evaluate_model(
        "ANN",
        y_test,
        ann_pred,
    ),

    evaluate_model(
        "Ensemble",
        y_test,
        ensemble_pred,
    ),
]


results_df = pd.DataFrame(
    results
)

# --------------------------------------------------
# SAVE MODELS AND PREPROCESSING
# --------------------------------------------------

joblib.dump(
    logistic,
    MODEL_DIR / "logistic_regression.joblib",
)

joblib.dump(
    random_forest,
    MODEL_DIR / "random_forest.joblib",
)

joblib.dump(
    scaler,
    MODEL_DIR / "standard_scaler.joblib",
)

# Save the exact feature order expected by the models.
joblib.dump(
    X.columns.tolist(),
    MODEL_DIR / "feature_columns.joblib",
)

# Save comparison metrics.
results_df.to_csv(
    RESULTS_DIR / "model_comparison.csv",
    index=False,
)

# Save the ANN in Keras format.
ann.save(
    MODEL_DIR / "ann_model.keras",
)

print("\nSaved models to:", MODEL_DIR)
print("Saved comparison results to:", RESULTS_DIR)
# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    ensemble_pred,
)

tn, fp, fn, tp = cm.ravel()


print()
print("=" * 65)
print("ENSEMBLE CONFUSION MATRIX")
print("=" * 65)

print()
print("                 Predicted")
print("                 0       1")

print(
    f"Actual 0      {tn:4d}    {fp:4d}"
)

print(
    f"Actual 1      {fn:4d}    {tp:4d}"
)

print()
print(f"True Negatives:   {tn}")
print(f"False Positives:  {fp}")
print(f"False Negatives:  {fn}")
print(f"True Positives:   {tp}")


# ============================================================
# MODEL COMPARISON
# ============================================================

print()
print("=" * 65)
print("MODEL COMPARISON")
print("=" * 65)

print(
    results_df.to_string(
        index=False,
        formatters={
            "Accuracy": "{:.3f}".format,
            "Precision": "{:.3f}".format,
            "Recall": "{:.3f}".format,
        },
    )
)


# ============================================================
# VOTING INFORMATION
# ============================================================

print()
print("=" * 65)
print("ENSEMBLE VOTING")
print("=" * 65)

print(
    "Logistic Regression predictions:",
    int(logistic_pred.sum()),
)

print(
    "Random Forest predictions:      ",
    int(rf_pred.sum()),
)

print(
    "ANN predictions:                ",
    int(ann_pred.sum()),
)

print(
    "Ensemble positive predictions:  ",
    int(ensemble_pred.sum()),
)