import pandas as pd

INPUT_FILE = "data/processed/california_acs_2010_2016_clean.csv"

df = pd.read_csv(INPUT_FILE)

# Housing-cost variable we'll initially investigate:
# B25088 = Median Selected Monthly Owner Costs
housing_2012 = "B25088_001E_2012"
housing_2016 = "B25088_001E_2016"

# Calculate change from 2012 to 2016
df["housing_cost_change"] = (
    df[housing_2016] - df[housing_2012]
)

# Calculate percentage change
df["housing_cost_pct_change"] = (
    df["housing_cost_change"] / df[housing_2012]
) * 100

print("Housing-cost change statistics:")
print(df["housing_cost_change"].describe())

print("\nHousing-cost percentage change statistics:")
print(df["housing_cost_pct_change"].describe())

print("\nMissing target values:")
print(df[
    ["GEOID", housing_2012, housing_2016,
     "housing_cost_change", "housing_cost_pct_change"]
].isna().sum())

print("\nLargest increases:")
print(
    df[
        ["GEOID", housing_2012, housing_2016,
         "housing_cost_change", "housing_cost_pct_change"]
    ]
    .sort_values("housing_cost_change", ascending=False)
    .head(10)
)

print("\nLargest decreases:")
print(
    df[
        ["GEOID", housing_2012, housing_2016,
         "housing_cost_change", "housing_cost_pct_change"]
    ]
    .sort_values("housing_cost_change")
    .head(10)
)
