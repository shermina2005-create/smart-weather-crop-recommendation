import pandas as pd

# Load weather dataset
df = pd.read_csv("data/weather/Cairo-Weather.csv")

# Convert date column
df["date"] = pd.to_datetime(df["date"])

# Sort by date
df = df.sort_values("date")

# Select important weather features
weather_data = df[
    [
        "date",
        "temperature_2m_mean",
        "relative_humidity_2m_mean",
        "precipitation_sum",
        "pressure_msl_mean",
    ]
]

# Remove missing values
weather_data = weather_data.dropna()

# Save prepared dataset
weather_data.to_csv("data/weather/weather_prepared.csv", index=False)

print("Weather data preparation completed!")
print("Rows:", len(weather_data))
print("Columns:", list(weather_data.columns))