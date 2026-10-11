import numpy as np
import pandas as pd
import geopandas as gpd
import requests
import zipfile
from pathlib import Path

from libpysal.weights import Queen

from preprocessing import census_api_key


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

DATA_DIR = Path("data/raw")
SHAPEFILE_ZIP = DATA_DIR / "tl_2010_06_tract10.zip"

MODEL_FILE = DATA_DIR / "model_dataset.csv"
OUTPUT_FILE = DATA_DIR / "model_dataset_spatial.csv"

S2503_2010_FILE = DATA_DIR / "s2503_2010_ca.csv"
S2503_2011_FILE = DATA_DIR / "s2503_2011_ca.csv"

TRACT_DIR = DATA_DIR / "tract10"


# ---------------------------------------------------------
# Census variables
# ---------------------------------------------------------

INCOME_VARIABLES = [
    "S2503_C01_002E",
    "S2503_C01_003E",
    "S2503_C01_004E",
    "S2503_C01_005E",
    "S2503_C01_006E",
    "S2503_C01_007E",
    "S2503_C01_008E",
    "S2503_C01_009E",
    "S2503_C01_010E",
    "S2503_C01_011E",
    "S2503_C01_012E",
]

HOUSING_VARIABLE = "B25105_001E"


# ---------------------------------------------------------
# Download S2503
# ---------------------------------------------------------

def download_s2503(year, output_file):

    if output_file.exists():
        print(f"{output_file} already exists. Skipping download.")
        return

    variables = [
        "NAME",
        *INCOME_VARIABLES,
    ]

    url = f"https://api.census.gov/data/{year}/acs/acs5/subject"

    params = {
        "get": ",".join(variables),
        "for": "tract:*",
        "in": "state:06",
        "key": census_api_key(),
    }

    print(f"Downloading S2503 for {year}...")

    # Retry because the Census server can reset large responses.
    for attempt in range(1, 6):

        try:

            response = requests.get(
                url,
                params=params,
                timeout=(30, 180),
            )

            response.raise_for_status()

            data = response.json()

            df = pd.DataFrame(
                data[1:],
                columns=data[0],
            )

            df.to_csv(
                output_file,
                index=False,
            )

            print(
                f"Saved {len(df):,} tracts to "
                f"{output_file}"
            )

            return

        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.ChunkedEncodingError,
            requests.exceptions.Timeout,
        ) as e:

            print(
                f"Attempt {attempt}/5 failed: "
                f"{type(e).__name__}"
            )

            if attempt < 5:
                import time
                wait = attempt * 3
                print(
                    f"Retrying in {wait} seconds..."
                )
                time.sleep(wait)

            else:
                raise


# ---------------------------------------------------------
# Load S2503 income distribution
# ---------------------------------------------------------

def load_income_data(year, filename):

    df = pd.read_csv(filename)

    df["state"] = df["state"].astype(str).str.zfill(2)
    df["county"] = df["county"].astype(str).str.zfill(3)
    df["tract"] = df["tract"].astype(str).str.zfill(6)

    df["tract_id"] = (
        df["state"]
        + df["county"]
        + df["tract"]
    )

    # Convert income-bin counts to numeric
    for col in INCOME_VARIABLES:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    for col in INCOME_VARIABLES:
        df.loc[
            df[col] < 0,
            col
        ] = float("nan")

    # Total occupied housing units represented by
    # the 11 income bins
    total = df[INCOME_VARIABLES].sum(axis=1)

    # Convert counts to probabilities
    for col in INCOME_VARIABLES:
        df[col] = df[col] / total

    return df[
        ["tract_id"] + INCOME_VARIABLES
    ]


# ---------------------------------------------------------
# Hellinger distance
# ---------------------------------------------------------

def hellinger_distance(row):

    distribution = row[INCOME_VARIABLES].to_numpy(
        dtype=float
    )

    # Baseline where everyone is in the
    # highest-income bin.
    affluent = [0.0] * 10 + [1.0]

    distance = (
        1 / (2 ** 0.5)
    ) * (
        (
            (
                distribution
                ** 0.5
            )
            -
            (
                pd.Series(affluent).to_numpy()
                ** 0.5
            )
        )
        ** 2
    ).sum() ** 0.5

    return distance


# ---------------------------------------------------------
# Calculate 2010 -> 2011 changes
# ---------------------------------------------------------

def build_observation_changes():

    print("\nBuilding 2010 -> 2011 changes...")

    income_2010 = load_income_data(
        2010,
        S2503_2010_FILE,
    )

    income_2011 = load_income_data(
        2011,
        S2503_2011_FILE,
    )

    income_2010["income_hellinger_2010"] = (
        income_2010.apply(
            hellinger_distance,
            axis=1,
        )
    )

    income_2011["income_hellinger_2011"] = (
        income_2011.apply(
            hellinger_distance,
            axis=1,
        )
    )

    income = income_2010[
        ["tract_id", "income_hellinger_2010"]
    ].merge(
        income_2011[
            ["tract_id", "income_hellinger_2011"]
        ],
        on="tract_id",
        how="inner",
    )

    income["income_change_2010_2011"] = (
        income["income_hellinger_2010"]
        - income["income_hellinger_2011"]
    )

    # -----------------------------------------------------
    # Housing
    # -----------------------------------------------------

    housing_2010 = pd.read_csv(
        DATA_DIR / "acs5_2010_ca.csv"
    )

    housing_2011 = pd.read_csv(
        DATA_DIR / "acs5_2011_ca.csv"
    )

    for df in [housing_2010, housing_2011]:

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

        df["tract_id"] = (
            df["state"]
            + df["county"]
            + df["tract"]
        )

        df[HOUSING_VARIABLE] = pd.to_numeric(
            df[HOUSING_VARIABLE],
            errors="coerce",
        )

        # Remove Census negative sentinel values
        df.loc[
            df[HOUSING_VARIABLE] < 0,
            HOUSING_VARIABLE
        ] = float("nan")

    housing = housing_2010[
        ["tract_id", HOUSING_VARIABLE]
    ].rename(
        columns={
            HOUSING_VARIABLE:
                "housing_cost_2010"
        }
    ).merge(
        housing_2011[
            ["tract_id", HOUSING_VARIABLE]
        ].rename(
            columns={
                HOUSING_VARIABLE:
                    "housing_cost_2011"
            }
        ),
        on="tract_id",
        how="inner",
    )

    housing["housing_change_2010_2011"] = (
        housing["housing_cost_2011"]
        - housing["housing_cost_2010"]
    )

    changes = housing[
        [
            "tract_id",
            "housing_change_2010_2011",
        ]
    ].merge(
        income[
            [
                "tract_id",
                "income_change_2010_2011",
            ]
        ],
        on="tract_id",
        how="inner",
    )

    print(
        f"Observation changes: "
        f"{len(changes):,} tracts"
    )

    return changes


# ---------------------------------------------------------
# Load Census tract geometry
# ---------------------------------------------------------

def load_tract_geometry():

    print("\nLoading Census tract geometry...")

    TRACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    shp_files = list(
        TRACT_DIR.glob("*.shp")
    )

    if not shp_files:

        print("Extracting tract shapefile...")

        with zipfile.ZipFile(
            SHAPEFILE_ZIP,
            "r",
        ) as z:
            z.extractall(TRACT_DIR)

        shp_files = list(
            TRACT_DIR.glob("*.shp")
        )

    if not shp_files:
        raise FileNotFoundError(
            "Could not find Census tract shapefile."
        )

    gdf = gpd.read_file(
        shp_files[0]
    )

    # 2010 TIGER/Line tract identifier
    if "GEOID10" in gdf.columns:

        gdf["tract_id"] = (
            gdf["GEOID10"]
            .astype(str)
            .str.zfill(11)
        )

    else:

        gdf["tract_id"] = (
            gdf["STATEFP10"].astype(str).str.zfill(2)
            + gdf["COUNTYFP10"].astype(str).str.zfill(3)
            + gdf["TRACTCE10"].astype(str).str.zfill(6)
        )

    print(
        f"Loaded {len(gdf):,} Census tracts"
    )

    return gdf[
        ["tract_id", "geometry"]
    ]


# ---------------------------------------------------------
# Calculate spatial features
# ---------------------------------------------------------

def _nanmean(values, index):
    """Mean of values[index] ignoring NaN; NaN if nothing is left."""
    if len(index) == 0:
        return float("nan")
    selected = values[index]
    selected = selected[~np.isnan(selected)]
    return float(selected.mean()) if selected.size else float("nan")


def calculate_spatial_features(
    gdf,
    changes,
):

    print("\nBuilding Queen contiguity weights...")

    # Keep EVERY tract polygon (left join) so contiguity is computed on
    # the real map. An inner join would delete tracts with missing data
    # and leave their neighbours with fewer (or no) neighbours.
    merged = gdf.merge(
        changes,
        on="tract_id",
        how="left",
    ).reset_index(drop=True)

    print(
        f"Tracts after geometry merge: "
        f"{len(merged):,}"
    )

    # Queen contiguity:
    # tracts touching along an edge OR corner
    weights = Queen.from_dataframe(
        merged,
        use_index=False,
    )

    # -----------------------------------------------------
    # Spatial lag
    # -----------------------------------------------------

    housing_values = (
        merged[
            "housing_change_2010_2011"
        ]
        .to_numpy(dtype=float)
    )

    income_values = (
        merged[
            "income_change_2010_2011"
        ]
        .to_numpy(dtype=float)
    )

    housing_lag = []
    income_lag = []

    for i in range(len(merged)):

        neighbors = list(
            weights.neighbors[i]
        )

        # Average only neighbours that have a value. A plain .mean()
        # returns NaN as soon as one neighbour is missing.
        housing_lag.append(
            _nanmean(housing_values, neighbors)
        )

        income_lag.append(
            _nanmean(income_values, neighbors)
        )

    merged["housing_spatial_lag"] = (
        housing_lag
    )

    merged["income_spatial_lag"] = (
        income_lag
    )

    # -----------------------------------------------------
    # Local Moran's I
    #
    # Paper's formula:
    #
    # I_i =
    # Z_i * (sum_j Z_j) /
    # (sum_j Z_j^2 / n)
    #
    # using adjacent tracts.
    # -----------------------------------------------------

    def local_moran(values):

        values = values.astype(float)

        # Only valid observations are used to calculate
        # the mean and variance.
        valid = ~pd.isna(values)

        mean = values[valid].mean()

        z = values - mean

        denominator = (
            (z[valid] ** 2).sum()
            / valid.sum()
        )

        result = []

        for i in range(len(values)):

            # Cannot calculate Moran's I for a tract
            # whose own value is missing.
            if pd.isna(values[i]):
                result.append(float("nan"))
                continue

            neighbors = list(
                weights.neighbors[i]
            )

            # Keep only neighbors with valid values.
            valid_neighbors = [
                j for j in neighbors
                if not pd.isna(values[j])
            ]

            if (
                len(valid_neighbors) == 0
                or denominator == 0
            ):
                result.append(float("nan"))

            else:

                neighbor_sum = z[
                    valid_neighbors
                ].sum()

                result.append(
                    z[i]
                    * neighbor_sum
                    / denominator
                )

        return result

    merged["housing_morans_i"] = (
        local_moran(housing_values)
    )

    merged["income_morans_i"] = (
        local_moran(income_values)
    )

    return merged[
        [
            "tract_id",
            "housing_spatial_lag",
            "income_spatial_lag",
            "housing_morans_i",
            "income_morans_i",
        ]
    ]


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("=" * 60)
    print("BUILDING SPATIAL FEATURES")
    print("=" * 60)

    # Download S2503
    download_s2503(
        2010,
        S2503_2010_FILE,
    )

    download_s2503(
        2011,
        S2503_2011_FILE,
    )

    # Calculate 2010 -> 2011 changes
    changes = build_observation_changes()

    # Load tract polygons
    gdf = load_tract_geometry()

    # Calculate spatial features
    spatial = calculate_spatial_features(
        gdf,
        changes,
    )

    # -----------------------------------------------------
    # Merge into existing model dataset
    # -----------------------------------------------------

    print("\nMerging spatial features...")

    model = pd.read_csv(
        MODEL_FILE
    )

    model["tract_id"] = (
        model["tract_id"]
        .astype(str)
        .str.zfill(11)
    )

    final = model.merge(
        spatial,
        on="tract_id",
        how="left",
    )

    final.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)

    print(
        f"Saved: {OUTPUT_FILE}"
    )

    print(
        f"Rows: {len(final):,}"
    )

    print(
        f"Columns: {len(final.columns):,}"
    )

    spatial_columns = [
        "housing_spatial_lag",
        "income_spatial_lag",
        "housing_morans_i",
        "income_morans_i",
    ]

    print("\nSpatial feature summary:")

    print(
        final[spatial_columns].describe()
    )

    print("\nMissing spatial values:")

    print(
        final[spatial_columns]
        .isna()
        .sum()
    )


if __name__ == "__main__":
    main()
