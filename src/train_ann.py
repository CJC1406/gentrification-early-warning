"""
ANN hyper-parameter tuning.

Configurations are compared on a VALIDATION split carved out of the
training data. The test set is touched once, for the chosen
configuration only. The winning configuration is written to
results/ann_best_config.json and picked up by train_ensemble.py.
"""

import json
import random

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from preprocessing import (
    DATA_DIR,
    RESULTS_DIR,
    SEED,
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

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CONFIGS = {
    "ANN Baseline": {"layers": [128, 64], "dropout": 0.30, "learning_rate": 0.001, "batch_size": 32},
    "ANN Low Dropout": {"layers": [128, 64], "dropout": 0.20, "learning_rate": 0.001, "batch_size": 32},
    "ANN Larger": {"layers": [256, 128, 64], "dropout": 0.20, "learning_rate": 0.001, "batch_size": 32},
    "ANN Lower LR": {"layers": [128, 64], "dropout": 0.20, "learning_rate": 0.0005, "batch_size": 32},
}


def build_ann(n_features, config):
    model = Sequential([Input(shape=(n_features,))])
    for units in config["layers"]:
        model.add(Dense(units, activation="relu"))
        model.add(Dropout(config["dropout"]))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(
        optimizer=Adam(learning_rate=config["learning_rate"]),
        loss="binary_crossentropy",
        metrics=["AUC"],
    )
    return model


def fit_ann(config, X_fit, y_fit, X_val, y_val):
    model = build_ann(X_fit.shape[1], config)
    model.fit(
        X_fit,
        y_fit,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=config["batch_size"],
        callbacks=[EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)],
        verbose=0,
    )
    return model


X, y, tract_ids = load_xy(DATA_DIR / "model_dataset.csv")
X_train, X_test, y_train, y_test, _, _ = split(X, y, tract_ids)

# Inner split: tuning happens only inside the training data.
X_fit, X_val, y_fit, y_val = train_test_split(
    X_train, y_train, test_size=0.20, random_state=SEED, stratify=y_train
)

prep = make_preprocessor(scale=True).fit(X_fit)
X_fit_s, X_val_s = prep.transform(X_fit), prep.transform(X_val)

val_rows = []
for name, config in CONFIGS.items():
    print(f"Training {name}...")
    model = fit_ann(config, X_fit_s, y_fit, X_val_s, y_val)
    prob = model.predict(X_val_s, verbose=0).ravel()
    val_rows.append(evaluate(name, y_val, (prob >= 0.5).astype(int), prob))

print("\nVALIDATION results (used for model selection)")
val_table = print_results(val_rows)

best_name = val_table.sort_values("ROC AUC", ascending=False).iloc[0]["Model"]
best_config = CONFIGS[best_name]
print(f"\nSelected by validation ROC AUC: {best_name} {best_config}")

with open(RESULTS_DIR / "ann_best_config.json", "w") as f:
    json.dump({"name": best_name, **best_config}, f, indent=2)

# One final evaluation on the untouched test set.
prep_full = make_preprocessor(scale=True).fit(X_train)
X_train_s, X_test_s = prep_full.transform(X_train), prep_full.transform(X_test)
X_tr, X_es, y_tr, y_es = train_test_split(
    X_train_s, y_train, test_size=0.20, random_state=SEED, stratify=y_train
)
final = fit_ann(best_config, X_tr, y_tr, X_es, y_es)
test_prob = final.predict(X_test_s, verbose=0).ravel()

print("\nTEST results for the selected configuration")
print_results(
    [
        majority_baseline(y_train, y_test)[0],
        evaluate(best_name, y_test, (test_prob >= 0.5).astype(int), test_prob),
    ]
)
