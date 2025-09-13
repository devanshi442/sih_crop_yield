# test_api.py
import requests

URL = "http://127.0.0.1:5000/predict"

# Example input data
payload = {
    "crop": "wheat",
    "soil_type": "loamy",
    "rainfall": 300,
    "temperature": 25
}

# Send POST request
response = requests.post(URL, json=payload)

# Print response
if response.status_code == 200:
    print("✅ Success! API Response:")
    print(response.json())
else:
    print("❌ Error:", response.status_code, response.text)
