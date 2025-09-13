import pandas as pd
import folium

def create_yield_map(csv_path: str, output_path: str):
    df = pd.read_csv(csv_path)
    df["Yield"] = df["Production"] / df["Area"]
    if "Latitude" not in df.columns or "Longitude" not in df.columns:
        raise ValueError("Latitude/Longitude columns missing")

    lat0, lon0 = df["Latitude"].mean(), df["Longitude"].mean()
    m = folium.Map(location=[lat0, lon0], zoom_start=6)

    for _, r in df.iterrows():
        folium.CircleMarker(
            location=[r["Latitude"], r["Longitude"]],
            radius=4,
            popup=f'{r.get("Crop","")}: Yield {r["Yield"]:.2f}',
            fill=True
        ).add_to(m)

    m.save(output_path)
    return output_path
