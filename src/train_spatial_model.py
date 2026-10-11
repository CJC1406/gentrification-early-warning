import pandas as pd

from pathlib import Path

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
)


DATA_DIR = Path("data/raw")


# ---------------------------------------------------------
# Load spatial dataset
# ---------------------------------------------------------

df = pd.read_csv(
    DATA_DIR / "model_dataset_spatial.csv"
)


# ---------------------------------------------------------
# Target
# ---------------------------------------------------------

y = df["gentrified"]


# ---------------------------------------------------------
# Remove target and non-feature columns
# ---------------------------------------------------------

X = df.drop(
    columns=[
        "tract_id",
        "housing_cost_change",
        "gentrified",
    ]
)


# ---------------------------------------------------------
# Convert everything to numeric
# ---------------------------------------------------------

X = X.apply(
    pd.to_numeric,
    errors="coerce",
)


# ---------------------------------------------------------
# Handle missing values
# ---------------------------------------------------------

X = X.fillna(
    X.median()
)


# ---------------------------------------------------------
# Train/test split
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

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


# ---------------------------------------------------------
# Random Forest
# ---------------------------------------------------------

model = RandomForestClassifier(

    n_estimators=125,

    max_features=18,

    random_state=42,

    n_jobs=-1,
)


model.fit(
    X_train,
    y_train,
)


# ---------------------------------------------------------
# Predictions
# ---------------------------------------------------------

predictions = model.predict(
    X_test
)


# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------

accuracy = accuracy_score(
    y_test,
    predictions,
)

precision = precision_score(
    y_test,
    predictions,
)

recall = recall_score(
    y_test,
    predictions,
)


print()

print(
    "Random Forest with Spatial Features"
)

print(
    "-----------------------------------"
)

print(
    f"Accuracy:  {accuracy:.3f}"
)

print(
    f"Precision: {precision:.3f}"
)

print(
    f"Recall:    {recall:.3f}"
)