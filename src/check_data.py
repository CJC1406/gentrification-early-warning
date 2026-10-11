import pandas as pd
import glob
import os

files = sorted(glob.glob("data/raw/acs_*.csv"))

for file in files:
    df = pd.read_csv(file)

    print("\n" + "=" * 50)
    print(os.path.basename(file))
    print("=" * 50)

    print("Rows:", len(df))
    print("Columns:", len(df.columns))
    print("Duplicate GEOIDs:", df["GEOID"].duplicated().sum())

    missing = df.isna().sum()
    missing = missing[missing > 0]

    if len(missing) == 0:
        print("Missing values: None")
    else:
        print("Missing values:")
        print(missing)
