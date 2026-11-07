
import os

import httpx

url = os.getenv("API_URL", "http://localhost:8080")
print("API_URL inside this container:", url)
for path in ("/health", "/docs"):
    try:
        r = httpx.get(url + path, timeout=10)
        print(path, r.status_code, r.text[:200].replace("\n", " "))
    except Exception as e:
        print(path, "FAILED:", type(e).__name__, e)