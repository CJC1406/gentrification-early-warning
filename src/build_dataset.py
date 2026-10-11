import pandas as pd
import os

INPUT_DIR = "data/raw"
OUTPUT_DIR = "data/processed"

os.makedirs(OUTPUT_DIR, exist_ok=True)

years = range(2010, 2017)

dataframes = []

for year in years:

    file = f"{INPUT_DIR}/acs_{year}.csv"

    print(f"Reading {file}...")

    df = pd.read_csv(file)

    # Remove columns that don't need to be repeated
    df = df.drop(columns=["NAME", "state", "county", "tract", "year"])

    # Rename every feature with its year
    rename_dict = {
        col: f"{col}_{year}"
        for col in df.columns
        if col != "GEOID"
    }

    df = df.rename(columns=rename_dict)

    dataframes.append(df)


# Start with the 2010 data
merged = dataframes[0]

# Add 2011–2016
for df in dataframes[1:]:

    merged = merged.merge(
        df,
        on="GEOID",
        how="inner",
        validate="one_to_one"
    )


output_file = f"{OUTPUT_DIR}/california_acs_2010_2016.csv"

merged.to_csv(output_file, index=False)

print("\nDataset created successfully.")
print("Shape:", merged.shape)
print("Saved to:", output_file)

print("\nMissing values:")
missing = merged.isna().sum()
print(missing[missing > 0])
