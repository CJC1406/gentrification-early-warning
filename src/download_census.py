import requests
import csv
from pathlib import Path

API_URL = "https://api.census.gov/data/{year}/acs/acs5"

YEARS = [2010, 2011, 2012, 2016]

VARIABLES = [
    "NAME",
    "B25105_001E",
]

OUTPUT_DIR = Path("data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def download_year(year):
    params = {
        "get": ",".join(VARIABLES),
        "for": "tract:*",
        "in": "state:06",
        "key": "da2c02abec9be8565b3b7c8d63ee324e99ad0f40",
    }

    response = requests.get(
        API_URL.format(year=year),
        params=params,
        timeout=60,
    )

    response.raise_for_status()

    data = response.json()

    header = data[0]
    rows = data[1:]

    output_file = OUTPUT_DIR / f"acs5_{year}_ca.csv"

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"{year}: downloaded {len(rows):,} tracts")
    print(f"      saved to {output_file}")


for year in YEARS:
    download_year(year)
