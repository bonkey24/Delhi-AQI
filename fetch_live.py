import requests

url = "https://delhi-aqi-4a51.onrender.com/predict?date=2026-04-13"
try:
    r = requests.get(url, timeout=10)
    print("Status Code:", r.status_code)
    print("Headers:", r.headers)
    print("Content (first 1000 chars):")
    print(r.text[:1000])
except Exception as e:
    print("Error querying URL:", e)
