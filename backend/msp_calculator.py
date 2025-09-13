def calculate_msp(crop: str, area: float, current_price: float) -> dict:
    crop = crop.title()  # Converts "wheat" → "Wheat" (case-insensitive)

    MSP_DICT = {
        "Wheat": 2300,
        "Rice": 2500,
        "Barley": 2100,
        "Maize": 2000,
        "Sugarcane": 2400,
        "Chickpea": 2200,
        "Arhar Dal": 2100,
        "Mustard": 2150,
        "Urad Daal": 2000
    }

    AVERAGE_YIELD = {
        "Wheat": 2.5,
        "Rice": 3.0,
        "Barley": 2.0,
        "Maize": 2.8,
        "Sugarcane": 60,
        "Chickpea": 1.8,
        "Arhar Dal": 2.0,
        "Mustard": 1.9,
        "Urad Daal": 2.1
    }

    if crop not in MSP_DICT:
        raise ValueError(f"MSP not available for crop: {crop}")

    yield_per_hectare = AVERAGE_YIELD[crop]
    expected_production = area * yield_per_hectare
    revenue_market = expected_production * current_price
    revenue_msp = expected_production * MSP_DICT[crop]

    recommendation = "Market price is good." if current_price >= MSP_DICT[crop] else "Consider selling at MSP or wait for price rise."

    return {
        "crop": crop,
        "area": area,
        "expected_production_qtl": expected_production,
        "current_market_price_per_qtl": current_price,
        "msp_price_per_qtl": MSP_DICT[crop],
        "revenue_at_market_price": revenue_market,
        "revenue_at_msp": revenue_msp,
        "recommendation": recommendation
    }
