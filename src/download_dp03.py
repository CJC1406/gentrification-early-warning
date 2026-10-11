import requests
import csv
from pathlib import Path

API_URL = "https://api.census.gov/data/{year}/acs/acs5/profile"

YEARS = [2010, 2011, 2012, 2016]

OUTPUT_DIR = Path("data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

API_KEY = "da2c02abec9be8565b3b7c8d63ee324e99ad0f40"


def get_variables(year):
    """
    Get all DP03 variables from the Census API.

    We keep both:
      - E  = estimate
      - PE = percentage estimate

    We exclude:
      - EA = annotation
      - PE variables are NOT accidentally excluded just because
        their names also end in 'E'.
    """

    url = (
        f"https://api.census.gov/data/"
        f"{year}/acs/acs5/profile/groups/DP03.json"
    )

    response = requests.get(
        url,
        params={"key": API_KEY},
        timeout=60,
    )

    response.raise_for_status()

    variables = response.json()["variables"]

    names = []

    for name in variables:

        # Keep percentage estimates
        if name.endswith("PE"):
            names.append(name)

        # Keep regular estimates, but not percentage estimates
        elif name.endswith("E") and not name.endswith("EA"):
            names.append(name)

    return sorted(names)


def download_year(year):

    print()
    print("=" * 60)
    print(f"Downloading DP03 for {year}")
    print("=" * 60)

    variables = get_variables(year)

    print(f"Variables found: {len(variables):,}")

    # Census API has URL-length limits.
    # Download variables in chunks.
    chunks = [
        variables[i:i + 40]
        for i in range(0, len(variables), 40)
    ]

    all_rows = None
    all_header = None

    for chunk_number, chunk in enumerate(chunks, start=1):

        print(
            f"Downloading chunk "
            f"{chunk_number}/{len(chunks)} "
            f"({len(chunk)} variables)"
        )

        params = {
            "get": ",".join(["NAME"] + chunk),
            "for": "tract:*",
            "in": "state:06",
            "key": API_KEY,
        }

        response = requests.get(
            API_URL.format(year=year),
            params=params,
            timeout=120,
        )

        # Show useful information if Census returns an error.
        if response.status_code != 200:
            print()
            print("Census API error")
            print("Status:", response.status_code)
            print("Response:", response.text[:1000])
            response.raise_for_status()

        try:
            data = response.json()
        except ValueError:
            print()
            print("Census API did not return JSON.")
            print("Status:", response.status_code)
            print("Response:", response.text[:1000])
            raise

        header = data[0]
        rows = data[1:]

        if all_rows is None:

            # First chunk contains:
            # NAME + variables + state + county + tract
            all_rows = {
                tuple(row[-3:]): row
                for row in rows
            }

            all_header = header

        else:

            for row in rows:

                tract_id = tuple(row[-3:])

                if tract_id not in all_rows:
                    continue

                existing = all_rows[tract_id]

                # Add variables from this chunk.
                #
                # row[0] = NAME
                # row[-3:] = state, county, tract
                # Therefore row[1:-3] = requested variables.
                existing.extend(row[1:-3])

            # Add only the new variable names.
            all_header.extend(header[1:-3])

    output_file = OUTPUT_DIR / f"dp03_{year}_ca.csv"

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(all_header)

        for row in all_rows.values():
            writer.writerow(row)

    print()
    print(f"{year}: {len(all_rows):,} tracts")
    print(f"Columns: {len(all_header):,}")
    print(f"Saved to: {output_file}")


def main():

    for year in YEARS:
        download_year(year)

    print()
    print("=" * 60)
    print("DP03 download complete")
    print("=" * 60)


if __name__ == "__main__":
    main() 