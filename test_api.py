import requests
url = "https://api.census.gov/data/2016/acs/acs5"
params = {
    "get": "NAME,B25105_001E",
    "for": "tract:*",
    "in": "state:06",
    "key": "da2c02abec9be8565b3b7c8d63ee324e99ad0f40"
}
r = requests.get(url, params=params, timeout=30)
print(r.status_code)
print(r.text[:1000])
