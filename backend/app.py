import sys
import os
import io
import json
from fastapi import FastAPI, HTTPException , Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
import pandas as pd
import matplotlib.pyplot as plt
import sys
import sys
import os

# Add parent folder to sys.path so Python can find model/
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
MODEL_DIR = os.path.join(ROOT, "model")
DATA_DIR = os.path.join(ROOT, "data")
FRONTEND_DIR = os.path.join(ROOT, "frontend")
sys.path.append(MODEL_DIR)

# CSV paths
YIELD_CSV = os.path.join(DATA_DIR, "yield_data.csv")
SATELLITE_CSV = os.path.join(DATA_DIR, "satellite_data.csv")
PEST_CSV = os.path.join(DATA_DIR, "pest_data.csv")

# -------- Import modules --------
from model.predict import predict_from_dict
from chatbot import get_chat_response
from farmer_profile import load_farmers, save_farmers
from geospatial import create_yield_map
from msp_calculator import calculate_msp
from ndvi_analysis import compute_ndvi
from pest_alerts import get_pest_alerts
from scenario_simulation import run_simulation

# -------- FastAPI app --------
app = FastAPI(title="Crop Yield Prediction Platform")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# -------- Startup --------
@app.on_event("startup")
def startup_event():
    print("Crop Yield Prediction FastAPI backend running!")

@app.get("/")
def home():
    return {"status": "ok", "message": "Crop Yield Prediction Platform running."}

# -------- Predict --------
@app.post("/predict")
def predict_endpoint(payload: dict = Body(...)):
    """
    Flexible predict endpoint: handles missing features by filling defaults.
    """
    import joblib
    import pandas as pd
    from fastapi import HTTPException

    model_type = payload.get("model", "rf")
    features = payload.get("features")
    if not isinstance(features, dict):
        raise HTTPException(status_code=400, detail="`features` must be a dictionary inside payload")

    # Load the pipeline model
    model_file = os.path.join(MODEL_DIR, f"model_{model_type}.pkl")
    try:
        model = joblib.load(model_file)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cannot load model: {e}")

    # Define defaults for missing features
    defaults = {
        "State": "Unknown",
        "District": "Unknown",
        "Season": "Kharif",
        "Area": 1.0,
        "Production": 1.0,
        "Rainfall": 0,
        "Temperature": 25,
        "Fertilizer": 0,
        "Pesticide": 0,
        "Latitude": 0.0,
        "Longitude": 0.0,
        "Crop": "Wheat",
        "Soil_pH": 6.5,
        "Nitrogen": 40,
        "Phosphorus": 30,
        "Potassium": 200,
        "Organic_Carbon": 0.7,
        "Humidity": 65,
        "WindSpeed": 10,
        "SunshineHours": 7,
        "Pest_Risk": "Low",
        "NDVI": 0.5
    }

    # Build a row for prediction
    row = {}
    for k in defaults.keys():
        row[k] = features.get(k, defaults[k])

    try:
        df = pd.DataFrame([row])
        val = model.predict(df)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    return {"predicted_yield": val, "model": model_type}


# -------- MSP --------
@app.get("/msp")
def msp_endpoint(crop: str, area: float, current_price: float):
    try:
        result = calculate_msp(crop.lower(), area, current_price)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return result

# -------- Farmer --------
from farmer_profile import load_farmers
@app.get("/farmer/history/{farmer_id}")
def get_farmer(farmer_id: str):
    farmers = load_farmers()
    if farmer_id not in farmers:
        raise HTTPException(status_code=404, detail="Farmer not found")
    return {"Farmer_ID": farmer_id, **farmers[farmer_id]}



# -------- Charts --------
@app.get("/chart/yield_by_crop")
def chart_yield_by_crop():
    if not os.path.exists(YIELD_CSV):
        raise HTTPException(status_code=404, detail="yield_data.csv not found")
    df = pd.read_csv(YIELD_CSV)
    df["Yield"] = df["Production"] / df["Area"]
    agg = df.groupby("Crop")["Yield"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8,5))
    agg.plot(kind="bar", ax=ax)
    ax.set_ylabel("Average Yield")
    ax.set_title("Average Yield by Crop")
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")

@app.get("/chart/rainfall_vs_yield")
def chart_rainfall_vs_yield():
    if not os.path.exists(YIELD_CSV):
        raise HTTPException(status_code=404, detail="yield_data.csv not found")
    df = pd.read_csv(YIELD_CSV)
    df["Yield"] = df["Production"] / df["Area"]
    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(df["Rainfall"], df["Yield"], alpha=0.6)
    ax.set_xlabel("Rainfall")
    ax.set_ylabel("Yield")
    ax.set_title("Rainfall vs Yield")
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")

# -------- Map --------
@app.get("/map/yield_map")
def yield_map():
    out_file = os.path.join(FRONTEND_DIR, "static_map.html")
    try:
        create_yield_map(YIELD_CSV, out_file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return FileResponse(out_file, media_type="text/html")

# -------- NDVI --------
@app.get("/ndvi")
def ndvi_endpoint():
    try:
        df = compute_ndvi(SATELLITE_CSV)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return df.to_dict(orient="records")

# -------- Pest Alerts --------
@app.get("/pest_alerts")
def pest_alerts():
    if not os.path.exists(PEST_CSV):
        raise HTTPException(status_code=404, detail="pest_data.csv not found")
    
    df = pd.read_csv(PEST_CSV)
    
    # Try both columns
    col = None
    for c in ["Pest_Level", "Pest_Risk"]:
        if c in df.columns:
            col = c
            break
    if not col:
        raise HTTPException(status_code=400, detail="pest_level column missing")
    
    # Numeric conversion
    if col == "Pest_Level":
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.dropna(subset=[col], inplace=True)
        alerts = df[df[col] > 5].to_dict(orient="records")
    else:
        # If it's categorical like "High/Medium/Low"
        alerts = df[df[col].str.lower() == "high"].to_dict(orient="records")
    
    return {"alerts": alerts}


# -------- Simulation --------
from fastapi import HTTPException, Body

@app.post("/simulate")
def simulate_endpoint(payload: dict = Body(...)):
    """
    Expects payload like:
    {
        "model": "rf",
        "scenario": {
            "Crop": "Arhar Dal",
            "State": "Uttar Pradesh",
            "District": "Kanpur",
            "Season": "Kharif",
            "Area": 200,
            "Production": 4500,
            "Rainfall": 100,
            "Temperature": 30,
            "Fertilizer": 50,
            "Pesticide": 10,
            "Latitude": 26.6,
            "Longitude": 80.3,
            ...
        }
    }
    """
    try:
        # Check payload
        if not payload or "scenario" not in payload:
            raise HTTPException(status_code=400, detail="No scenario provided")
        scenario = payload["scenario"]
        if not isinstance(scenario, dict):
            raise HTTPException(status_code=400, detail="`scenario` must be a dictionary")

        model_type = payload.get("model", "rf")
        
        # Load model
        import joblib
        import os
        model_file = os.path.join(MODEL_DIR, f"model_{model_type}.pkl")
        if not os.path.exists(model_file):
            raise HTTPException(status_code=404, detail=f"Model file not found for {model_type}")
        model = joblib.load(model_file)

        # Filter features to what model expects
        model_features = model.feature_names_in_
        filtered_features = {k: scenario[k] for k in model_features if k in scenario}

        missing = [f for f in model_features if f not in filtered_features]
        if missing:
            raise HTTPException(status_code=400, detail=f"Missing features for model: {missing}")

        # Make prediction
        import pandas as pd
        df = pd.DataFrame([filtered_features])
        prediction = model.predict(df)[0]

        return {"predicted_yield": prediction, "scenario": filtered_features}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")



# -------- Chat --------
@app.post("/chat")
def chat_endpoint(payload: dict = Body(...)):
    message = payload.get("message", "")
    if not message:
        raise HTTPException(status_code=400, detail="Message is empty")
    reply = get_chat_response(message)
    return {"response": reply}
