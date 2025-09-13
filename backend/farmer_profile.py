import os
import json

FARMER_STORE = "data/farmers.json"

def ensure_store():
    if not os.path.exists("data"):
        os.makedirs("data")
    if not os.path.exists(FARMER_STORE):
        with open(FARMER_STORE, "w") as f:
            json.dump({}, f)

def load_farmers() -> dict[int, dict]:
    ensure_store()
    with open(FARMER_STORE, "r") as f:
        data = json.load(f)
    # Convert string keys to integers
    return {int(k): v for k, v in data.items()}


def save_farmers(data: dict):
    ensure_store()
    with open(FARMER_STORE, "w") as f:
        json.dump(data, f, indent=2)
