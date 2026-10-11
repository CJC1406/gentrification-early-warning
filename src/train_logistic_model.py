"""
Logistic regression on the 2010/2011 DP03 features.

The regularisation strength C is chosen by 5-fold cross-validation on
the TRAINING set only, so the test set is used exactly once.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
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

pipeline = Pipeline(
    [
        ("prep", make_preprocessor(scale=True)),
        ("lr", LogisticRegression(max_iter=2000, random_state=SEED)),
    ]
)

search = GridSearchCV(
    pipeline,
    param_grid={"lr__C": [0.001, 0.005, 0.01, 0.05, 0.1, 1.0]},
    scoring="roc_auc",
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
    n_jobs=-1,
)
search.fit(X_train, y_train)

print(f"\nBest C (training-set CV): {search.best_params_['lr__C']}")
print(f"Cross-validated ROC AUC:  {search.best_score_:.3f}")

model = search.best_estimator_
pred = model.predict(X_test)
prob = model.predict_proba(X_test)[:, 1]

baseline, _, _ = majority_baseline(y_train, y_test)

print("\nLogistic Regression results (held-out test set)")
print_results([baseline, evaluate("Logistic Regression", y_test, pred, prob)])
