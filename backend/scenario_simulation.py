import pandas as pd
from model.predict import predict_from_dict  # import your predict helper

def run_simulation(model_type: str, scenarios: list):
    results = []
    for feat_dict in scenarios:
        try:
            predicted_yield = predict_from_dict(model_type, feat_dict)
            results.append({"features": feat_dict, "predicted_yield": predicted_yield})
        except Exception as e:
            results.append({"features": feat_dict, "error": str(e)})
    return results
