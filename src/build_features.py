import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/raw")


def load_dp03(year):
    """
    Load a DP03 Census file and create a stable tract ID.
    """

    df = pd.read_csv(
        DATA_DIR / f"dp03_{year}_ca.csv"
    )

    # Force geographic identifiers to strings
    df["state"] = (
        df["state"]
        .astype(str)
        .str.zfill(2)
    )

    df["county"] = (
        df["county"]
        .astype(str)
        .str.zfill(3)
    )

    df["tract"] = (
        df["tract"]
        .astype(str)
        .str.zfill(6)
    )

    # Create stable 11-digit Census tract ID
    df["tract_id"] = (
        df["state"]
        + df["county"]
        + df["tract"]
    )

    return df


# ============================================================
# LOAD CENSUS DATA
# ============================================================

df2010 = load_dp03(2010)
df2011 = load_dp03(2011)


# ============================================================
# REMOVE MALFORMED CENSUS PERCENTAGE VARIABLES
# ============================================================

# These variables are officially percentage variables (PE),
# but the 2010/2011 Census API response returns the same
# integer values as their corresponding E variables.
#
# Example:
# DP03_0089E  = 276
# DP03_0089PE = 276
#
# Since 276 cannot be a percentage, these malformed PE
# variables are removed rather than incorrectly recalculated.

BAD_PE_COLUMNS = [
    "DP03_0001PE",
    "DP03_0008PE",
    "DP03_0010PE",
    "DP03_0014PE",
    "DP03_0016PE",
    "DP03_0018PE",
    "DP03_0026PE",
    "DP03_0032PE",
    "DP03_0046PE",
    "DP03_0051PE",
    "DP03_0075PE",
    "DP03_0089PE",
]

df2010 = df2010.drop(
    columns=BAD_PE_COLUMNS,
    errors="ignore",
)

df2011 = df2011.drop(
    columns=BAD_PE_COLUMNS,
    errors="ignore",
)


# ============================================================
# LOAD TARGET
# ============================================================

target = pd.read_csv(
    DATA_DIR / "housing_cost_target.csv"
)

# Force target tract ID to string
target["tract_id"] = (
    target["tract_id"]
    .astype(str)
    .str.zfill(11)
)


# ============================================================
# REMOVE GEOGRAPHIC COLUMNS
# ============================================================

geo_columns = [
    "NAME",
    "state",
    "county",
    "tract",
]

df2010 = df2010.drop(
    columns=geo_columns,
    errors="ignore",
)

df2011 = df2011.drop(
    columns=geo_columns,
    errors="ignore",
)


# ============================================================
# RENAME FEATURES BY YEAR
# ============================================================

df2010 = df2010.rename(
    columns={
        col: f"{col}_2010"
        for col in df2010.columns
        if col != "tract_id"
    }
)

df2011 = df2011.rename(
    columns={
        col: f"{col}_2011"
        for col in df2011.columns
        if col != "tract_id"
    }
)


# ============================================================
# COMBINE 2010 AND 2011 FEATURES
# ============================================================

features = df2010.merge(
    df2011,
    on="tract_id",
    how="inner",
)


# ============================================================
# ADD TARGET
# ============================================================

features = features.merge(
    target[
        [
            "tract_id",
            "housing_cost_change",
            "gentrified",
        ]
    ],
    on="tract_id",
    how="inner",
)


# ============================================================
# SAVE MODEL DATASET
# ============================================================

output = DATA_DIR / "model_dataset.csv"

features.to_csv(
    output,
    index=False,
)


# ============================================================
# PRINT DATASET SUMMARY
# ============================================================

print()
print("=" * 60)
print("MODEL DATASET CREATED")
print("=" * 60)

print(f"Saved: {output}")
print(f"Rows: {len(features):,}")
print(f"Columns: {len(features.columns):,}")

print()

print("Target distribution:")
print(features["gentrified"].value_counts())

print()

print("Feature count:")

feature_columns = [
    col
    for col in features.columns
    if col not in [
        "tract_id",
        "housing_cost_change",
        "gentrified",
    ]
]

print(f"{len(feature_columns):,}")

print()

print("Removed malformed PE variables:")
print(f"{len(BAD_PE_COLUMNS)}")

print()

print("Dataset creation complete.")