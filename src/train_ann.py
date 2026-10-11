import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam


DATA_DIR = Path("data/raw")


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

df = pd.read_csv(DATA_DIR / "model_dataset.csv")

y = df["gentrified"]

X = df.drop(
    columns=[
        "tract_id",
        "housing_cost_change",
        "gentrified",
    ]
)

X = X.apply(pd.to_numeric, errors="coerce")

# Census missing-value sentinel
X = X.replace(999999999, np.nan)

# Median imputation
X = X.fillna(X.median())

# Safety net
X = X.fillna(0)


# --------------------------------------------------
# Train/test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)


print(f"Training samples: {len(X_train):,}")
print(f"Test samples:     {len(X_test):,}")
print(f"Features:         {X.shape[1]:,}")


# --------------------------------------------------
# Standardize
# --------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# --------------------------------------------------
# ANN configurations
# --------------------------------------------------

configs = {

    "ANN Baseline": {
        "layers": [128, 64],
        "dropout": 0.30,
        "learning_rate": 0.001,
        "batch_size": 32,
    },

    "ANN Low Dropout": {
        "layers": [128, 64],
        "dropout": 0.20,
        "learning_rate": 0.001,
        "batch_size": 32,
    },

    "ANN Larger": {
        "layers": [256, 128, 64],
        "dropout": 0.20,
        "learning_rate": 0.001,
        "batch_size": 32,
    },

    "ANN Lower LR": {
        "layers": [128, 64],
        "dropout": 0.20,
        "learning_rate": 0.0005,
        "batch_size": 32,
    },
}


results = []


# --------------------------------------------------
# Train each configuration
# --------------------------------------------------

for name, config in configs.items():

    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)

    model = Sequential()

    # Input layer
    model.add(
        Input(shape=(X_train_scaled.shape[1],))
    )

    # Hidden layers
    for neurons in config["layers"]:

        model.add(
            Dense(
                neurons,
                activation="relu"
            )
        )

        model.add(
            Dropout(config["dropout"])
        )

    # Output layer
    model.add(
        Dense(
            1,
            activation="sigmoid"
        )
    )

    # Compile
    model.compile(
        optimizer=Adam(
            learning_rate=config["learning_rate"]
        ),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    # Early stopping
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
    )

    # Train
    model.fit(
        X_train_scaled,
        y_train,
        validation_split=0.20,
        epochs=100,
        batch_size=config["batch_size"],
        callbacks=[early_stopping],
        verbose=0,
    )

    # Predictions
    probabilities = model.predict(
        X_test_scaled,
        verbose=0
    )

    predictions = (
        probabilities >= 0.5
    ).astype(int).flatten()

    # Metrics
    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
    })


# --------------------------------------------------
# Results
# --------------------------------------------------

results_df = pd.DataFrame(results)

print("\n")
print("=" * 60)
print("ANN TUNING RESULTS")
print("=" * 60)

print(
    results_df.to_string(
        index=False,
        formatters={
            "Accuracy": "{:.3f}".format,
            "Precision": "{:.3f}".format,
            "Recall": "{:.3f}".format,
        }
    )
)