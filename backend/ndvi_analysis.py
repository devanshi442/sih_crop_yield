import pandas as pd

def compute_ndvi(csv_path: str) -> pd.DataFrame:
    if not csv_path or not pd.io.common.file_exists(csv_path):
        raise FileNotFoundError("satellite_data.csv not found")

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.lower()  # lowercase columns
    if "nir" not in df.columns or "red" not in df.columns:
        raise ValueError("NIR and RED columns required")

    df["nir"] = pd.to_numeric(df["nir"], errors="coerce")
    df["red"] = pd.to_numeric(df["red"], errors="coerce")
    df.dropna(subset=["nir","red"], inplace=True)

    df["ndvi"] = (df["nir"] - df["red"]) / (df["nir"] + df["red"])
    return df[["latitude","longitude","ndvi"]] if "latitude" in df.columns and "longitude" in df.columns else df
