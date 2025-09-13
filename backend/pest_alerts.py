import os
import pandas as pd
from fastapi import HTTPException

def get_pest_alerts(csv_path: str):
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail="pest_data.csv not found")

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.lower()

    if "pest_level" not in df.columns:
        raise HTTPException(status_code=400, detail="pest_level column missing")

    df["pest_level"] = pd.to_numeric(df["pest_level"], errors="coerce")
    df.dropna(subset=["pest_level"], inplace=True)

    alerts = df[df["pest_level"] > 5].to_dict(orient="records")
    return alerts
