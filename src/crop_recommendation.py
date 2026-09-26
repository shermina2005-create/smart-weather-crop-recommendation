def recommend_crop(temperature, humidity, rainfall, land_type):
    """
    Simple rule-based crop recommendation.
    """

    if land_type == "AnnualCrop":
        if rainfall >= 100 and humidity >= 60:
            return "Rice"
        elif temperature >= 20 and temperature <= 30:
            return "Wheat"
        else:
            return "Maize"

    elif land_type == "HerbaceousVegetation":
        if temperature >= 20 and rainfall >= 50:
            return "Maize"
        else:
            return "Wheat"

    elif land_type == "Forest":
        return "Agroforestry / Suitable plantation crops"

    elif land_type == "River":
        return "Rice"

    elif land_type == "Residential":
        return "Not suitable for crop cultivation"

    else:
        return "Maize"


# Example test
temperature = 28
humidity = 65
rainfall = 120
land_type = "AnnualCrop"

crop = recommend_crop(
    temperature,
    humidity,
    rainfall,
    land_type
)

print("Crop Recommendation")
print("-------------------")
print("Recommended Crop:", crop)