
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("data/raw")


def load_housing(year):
    path = DATA_DIR / f"acs5_{year}_ca.csv"

    df = pd.read_csv(
        path,
        dtype={
            "state": str,
            "county": str,
            "tract": str,
        },
    )

    # Preserve Census geographic identifiers.
    for col, width in [
        ("state", 2),
        ("county", 3),
        ("tract", 6),
    ]:
        df[col] = df[col].str.zfill(width)

    df["tract_id"] = (
        df["state"] + df["county"] + df["tract"]
    )

    cost_col = "B25105_001E"

    df[cost_col] = pd.to_numeric(
        df[cost_col], errors="coerce"
    )

    return df[["tract_id", cost_col]].rename(
        columns={cost_col: f"housing_cost_{year}"}
    )


# Load the two years used to define the historical target.
df2012 = load_housing(2012)
df2016 = load_housing(2016)

# Ensure one record per tract in each year.
df2012 = df2012.drop_duplicates("tract_id")
df2016 = df2016.drop_duplicates("tract_id")

df = df2012.merge(
    df2016,
    on="tract_id",
    how="inner",
    validate="one_to_one",
)

before = len(df)

# Census missing-value sentinels are negative numeric values.
# Housing costs must be positive to be usable.
for col in ["housing_cost_2012", "housing_cost_2016"]:
    df[col] = df[col].replace(
        [np.inf, -np.inf], np.nan
    )
    df.loc[df[col] <= 0, col] = np.nan

# Both years must have valid observations.
df = df.dropna(
    subset=["housing_cost_2012", "housing_cost_2016"]
).copy()

df["housing_cost_change"] = (
    df["housing_cost_2016"] - df["housing_cost_2012"]
)

# This is a housing-cost-increase proxy, not a complete
# definition of gentrification.
df["gentrified"] = (
    df["housing_cost_change"] > 0
).astype(int)

output = DATA_DIR / "housing_cost_target.csv"
df.to_csv(output, index=False)

print(f"Saved: {output}")
print(f"Matched tracts before validation: {before:,}")
print(f"Valid tracts after validation: {len(df):,}")
print(f"Excluded tracts: {before - len(df):,}")

print("\nHousing costs:")
print(
    df[
        ["housing_cost_2012", "housing_cost_2016",
         "housing_cost_change"]
    ].describe().round(2)
)

print("\nTarget-label distribution:")
print(df["gentrified"].value_counts().sort_index())

print("\nInvalid values remaining:")
print(
    (df[["housing_cost_2012", "housing_cost_2016"]] <= 0)
    .sum()
)
