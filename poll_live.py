import requests
import time

url = "https://delhi-aqi-4a51.onrender.com/predict?date=2026-04-13"
print("Polling Render deployment... waiting for the new code to go live...")

start_time = time.time()
timeout = 300  # 5 minutes

while time.time() - start_time < timeout:
    try:
        r = requests.get(url, timeout=10)
        content_type = r.headers.get("Content-Type", "")
        print(f"[{int(time.time() - start_time)}s] Status: {r.status_code} | Content-Type: {content_type}")
        
        if "application/json" in content_type:
            print("SUCCESS! The new container is live and returned JSON!")
            print("Response payload:")
            print(r.text)
            break
        elif "html" not in content_type.lower():
            print("Received unexpected content type:")
            print(r.text[:500])
            break
    except Exception as e:
        print("Polling error:", e)
    
    time.sleep(15)

if time.time() - start_time >= timeout:
    print("Polling timed out. The new container has not gone live yet or is stuck.")
