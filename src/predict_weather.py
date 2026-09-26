import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# Load prepared weather data
df = pd.read_csv("data/weather/weather_prepared.csv")

features = [
    "temperature_2m_mean",
    "relative_humidity_2m_mean",
    "precipitation_sum",
    "pressure_msl_mean"
]

data = df[features].values

# Scale data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Load trained LSTM model
model = load_model("models/weather_lstm.keras")

# Use the last 7 days as input
sequence = scaled_data[-7:]
X_input = np.array([sequence])

# Predict the next day
prediction_scaled = model.predict(X_input, verbose=0)

# Convert prediction back to original values
prediction = scaler.inverse_transform(prediction_scaled)

print("\nNext Day Weather Prediction")
print("---------------------------")
print(f"Temperature: {prediction[0][0]:.2f} °C")
print(f"Humidity: {prediction[0][1]:.2f} %")
print(f"Precipitation: {prediction[0][2]:.2f} mm")
print(f"Pressure: {prediction[0][3]:.2f} hPa")