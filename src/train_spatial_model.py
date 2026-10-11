"""
Random Forest with vs. without spatial features.

Both models use the same rows, the same split and the same settings, so
the difference in scores is attributable to the spatial features.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from preprocessing import (
    DATA_DIR,
    SEED,
    evaluate,
    load_xy,
    majority_baseline,
    make_preprocessor,
    print_results,
    split,
)

SPATIAL_COLUMNS = [
    "housing_spatial_lag",
    "income_spatial_lag",
    "housing_morans_i",
    "income_morans_i",
]

X, y, tract_ids = load_xy(DATA_DIR / "model_dataset_spatial.csv")
X_train, X_test, y_train, y_test, _, _ = split(X, y, tract_ids)

print(f"Training samples: {len(X_train):,}")
print(f"Test samples:     {len(X_test):,}")
print(f"Features:         {X.shape[1]:,}")


def random_forest():
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


rows = [majority_baseline(y_train, y_test)[0]]

spatial = [c for c in SPATIAL_COLUMNS if c in X.columns]

for name, columns in [
    ("Random Forest (no spatial)", [c for c in X.columns if c not in spatial]),
    ("Random Forest (with spatial)", list(X.columns)),
]:
    model = random_forest()
    model.fit(X_train[columns], y_train)
    pred = model.predict(X_test[columns])
    prob = model.predict_proba(X_test[columns])[:, 1]
    rows.append(evaluate(name, y_test, pred, prob))

print("\nSpatial-feature comparison (held-out test set)")
print_results(rows)
