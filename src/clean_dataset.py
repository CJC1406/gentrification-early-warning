import pandas as pd
import os

INPUT_FILE = "data/processed/california_acs_2010_2016.csv"
OUTPUT_FILE = "data/processed/california_acs_2010_2016_clean.csv"

df = pd.read_csv(INPUT_FILE)

# Census special value meaning data are unavailable/not applicable
SPECIAL_VALUE = -666666666

# Replace the special Census value with missing data
df = df.replace(SPECIAL_VALUE, pd.NA)

# Convert numeric columns back to numeric
for col in df.columns:
    if col != "GEOID":
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Save cleaned dataset
df.to_csv(OUTPUT_FILE, index=False)

print("Cleaned dataset created.")
print("Shape:", df.shape)
print("Saved to:", OUTPUT_FILE)

print("\nRemaining missing values:")
missing = df.isna().sum()
print(missing[missing > 0])
