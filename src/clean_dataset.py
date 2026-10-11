import pandas as pd

from preprocessing import clean_census_values

INPUT_FILE = "data/processed/california_acs_2010_2016.csv"
OUTPUT_FILE = "data/processed/california_acs_2010_2016_clean.csv"

df = pd.read_csv(INPUT_FILE)

# Convert to numeric and replace ALL Census sentinel codes
# (-666666666, -999999999, -888888888, ...) with missing values.
value_cols = [col for col in df.columns if col != "GEOID"]
df[value_cols] = clean_census_values(df[value_cols])

# Save cleaned dataset
df.to_csv(OUTPUT_FILE, index=False)

print("Cleaned dataset created.")
print("Shape:", df.shape)
print("Saved to:", OUTPUT_FILE)

print("\nRemaining missing values:")
missing = df.isna().sum()
print(missing[missing > 0])
