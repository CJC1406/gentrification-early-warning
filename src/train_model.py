"""Random Forest on the 2010/2011 DP03 features."""

import pandas as pd

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

X, y, tract_ids = load_xy(DATA_DIR / "model_dataset.csv")
X_train, X_test, y_train, y_test, _, _ = split(X, y, tract_ids)

print(f"Training samples: {len(X_train):,}")
print(f"Test samples:     {len(X_test):,}")
print(f"Features:         {X.shape[1]:,}")
print(f"Positive rate (train): {y_train.mean():.3f}")

model = Pipeline(
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

model.fit(X_train, y_train)

importance = pd.Series(
    model.named_steps["rf"].feature_importances_,
    index=X.columns,
).sort_values(ascending=False)

print("\nTop 20 feature importances")
print("--------------------------")
print(importance.head(20).to_string())

pred = model.predict(X_test)
prob = model.predict_proba(X_test)[:, 1]

baseline, _, _ = majority_baseline(y_train, y_test)

print("\nRandom Forest results (held-out test set)")
print_results([baseline, evaluate("Random Forest", y_test, pred, prob)])
