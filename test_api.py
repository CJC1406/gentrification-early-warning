import os

import requests

url = "https://api.census.gov/data/2016/acs/acs5"
params = {
    "get": "NAME,B25105_001E",
    "for": "tract:*",
    "in": "state:06",
    "key": os.environ["CENSUS_API_KEY"],
}
r = requests.get(url, params=params, timeout=30)
print(r.status_code)
print(r.text[:1000])
