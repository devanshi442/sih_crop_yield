"""
Train models from CSV files:
- Expects data/yield_data.csv, data/soil_data.csv, data/weather_data.csv, (optional) data/farmer_history.csv
- Produces model/model_rf.pkl and model/model_xgb.pkl
Features used: State, District, Crop, Season, Area, Production, Rainfall, Fertilizer, Pesticide, Latitude, Longitude
Target: Yield = Production / Area
"""
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import xgboost as xg
from sklearn.metrics import mean_squared_error, r2_score

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
DATA_DIR = os.path.join(ROOT, "data")
MODEL_DIR = os.path.join(ROOT, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

# Expected files
YIELD_CSV = os.path.join(DATA_DIR, "yield_data.csv")
SOIL_CSV = os.path.join(DATA_DIR, "soil_data.csv")
WEATHER_CSV = os.path.join(DATA_DIR, "weather_data.csv")
FARMER_CSV = os.path.join(DATA_DIR, "farmer_history.csv")  # optional

CATEGORICAL = ["State", "District", "Crop", "Season"]
NUMERICAL = ["Area", "Production", "Rainfall", "Fertilizer", "Pesticide", "Latitude", "Longitude"]
ALL_FEATURES = CATEGORICAL + NUMERICAL

def load_and_merge():
    if not os.path.exists(YIELD_CSV):
        raise FileNotFoundError(f"Missing {YIELD_CSV}")
    ydf = pd.read_csv(YIELD_CSV)
    # Merge soil and weather if available. We assume common keys: Region, Year or State/District
    df = ydf.copy()
    if os.path.exists(SOIL_CSV):
        soil = pd.read_csv(SOIL_CSV)
        common = [k for k in ["State", "District", "Region", "Year"] if k in soil.columns and k in df.columns]
        if common:
            df = df.merge(soil, on=common, how="left")
    if os.path.exists(WEATHER_CSV):
        weather = pd.read_csv(WEATHER_CSV)
        common = [k for k in ["State", "District", "Region", "Year"] if k in weather.columns and k in df.columns]
        if common:
            df = df.merge(weather, on=common, how="left")
    return df

def preprocess_and_train():
    df = load_and_merge()
    # Ensure required columns present
    missing = [c for c in ["Area", "Production"] if c not in df.columns]
    if missing:
        raise ValueError(f"yield_data.csv must contain columns: {missing}")

    # compute Yield
    df["Yield"] = df["Production"] / df["Area"]
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=["Yield"], inplace=True)

    # fill missing numeric values with median (future-proof style)
    for col in NUMERICAL:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = 0.0

    # fill categorical with 'Unknown'
    for col in CATEGORICAL:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str)
        else:
            df[col] = "Unknown"

    X = df[ALL_FEATURES]
    y = df["Yield"]

    # split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    # preprocessing
    preprocessor = ColumnTransformer(transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ("num", StandardScaler(), NUMERICAL)
    ], remainder="drop")

    # RandomForest pipeline
    rf_pipeline = Pipeline([
        ("pre", preprocessor),
        ("model", RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))
    ])

    print("Training RandomForest...")
    rf_pipeline.fit(X_train, y_train)
    rf_preds = rf_pipeline.predict(X_test)
    rf_rmse = mean_squared_error(y_test, rf_preds) ** 0.5
    print("RF RMSE:", rf_rmse, "R2:", r2_score(y_test, rf_preds))
    joblib.dump(rf_pipeline, os.path.join(MODEL_DIR, "model_rf.pkl"))

    # XGBoost pipeline
    xgb_pipeline = Pipeline([
        ("pre", preprocessor),
        ("model", xg.XGBRegressor(n_estimators=300, learning_rate=0.08, random_state=42, n_jobs=-1))
    ])

    print("Training XGBoost...")
    xgb_pipeline.fit(X_train, y_train)
    xgb_preds = xgb_pipeline.predict(X_test)
    xgb_rmse = mean_squared_error(y_test, xgb_preds) ** 0.5
    print("XGB RMSE:", xgb_rmse, "R2:", r2_score(y_test, xgb_preds))
    joblib.dump(xgb_pipeline, os.path.join(MODEL_DIR, "model_xgb.pkl"))

    print("Models saved to", MODEL_DIR)

if __name__ == "__main__":
    preprocess_and_train()
