import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load prepared weather data
df = pd.read_csv("data/weather/weather_prepared.csv")

# Select features for prediction
features = [
    "temperature_2m_mean",
    "relative_humidity_2m_mean",
    "precipitation_sum",
    "pressure_msl_mean"
]

data = df[features].values

# Scale data between 0 and 1
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Create sequences for LSTM
sequence_length = 7

X = []
y = []

for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i-sequence_length:i])
    y.append(scaled_data[i])

X = np.array(X)
y = np.array(y)

# Build LSTM model
model = Sequential([
    LSTM(64, input_shape=(X.shape[1], X.shape[2])),
    Dense(32, activation="relu"),
    Dense(4)
])

model.compile(
    optimizer="adam",
    loss="mse"
)

print("Training LSTM model...")

# Train the model
model.fit(
    X,
    y,
    epochs=10,
    batch_size=32,
    validation_split=0.2
)

# Save the trained model
model.save("models/weather_lstm.keras")

print("LSTM model training completed!")
print("Model saved to models/weather_lstm.keras")