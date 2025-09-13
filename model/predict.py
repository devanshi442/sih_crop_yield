"""
Load trained model and predict. Exposes a helper function predict_from_dict(features, model='rf')
Assumes features dict keys in the exact order of ALL_FEATURES used in training
"""
import os
import joblib
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE)

RF_PATH = os.path.join(MODEL_DIR, "model_rf.pkl")
XGB_PATH = os.path.join(MODEL_DIR, "model_xgb.pkl")

if not os.path.exists(RF_PATH) or not os.path.exists(XGB_PATH):
    print("Warning: model files not found in model/. Train models first with model/train_model.py")

_rf = joblib.load(RF_PATH) if os.path.exists(RF_PATH) else None
_xgb = joblib.load(XGB_PATH) if os.path.exists(XGB_PATH) else None

def predict_from_dict(features: dict, model="rf"):
    """
    features: dict with keys: State, District, Crop, Season, Area, Production, Rainfall, Fertilizer, Pesticide, Latitude, Longitude
    model: 'rf' or 'xgb'
    returns float predicted Yield
    """
    # maintain consistent feature order
    order = ["State", "District", "Crop", "Season", "Area", "Production", "Rainfall", "Fertilizer", "Pesticide", "Latitude", "Longitude"]
    X = [features.get(k, 0.0) for k in order]
    X_arr = np.array([X])
    if model == "rf":
        if _rf is None:
            raise RuntimeError("RF model not loaded")
        return float(_rf.predict(X_arr)[0])
    elif model == "xgb":
        if _xgb is None:
            raise RuntimeError("XGB model not loaded")
        return float(_xgb.predict(X_arr)[0])
    else:
        raise ValueError("model should be 'rf' or 'xgb'")
