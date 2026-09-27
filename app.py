import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path

from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
from sklearn.preprocessing import MinMaxScaler


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Smart Weather & Crop Recommendation",
    page_icon="🌱",
    layout="wide"
)


# -----------------------------
# Title
# -----------------------------
st.title("🌱 Smart Weather & Crop Recommendation")
st.subheader("Satellite-Based Weather Forecasting System")

st.write(
    "An AI-based system for weather prediction, satellite image classification, "
    "and intelligent crop recommendation."
)

st.divider()


# -----------------------------
# Load Weather Model
# -----------------------------
@st.cache_resource
def load_weather_model():
    return load_model("models/weather_lstm.keras")


# -----------------------------
# Load Satellite CNN
# -----------------------------
@st.cache_resource
def load_satellite_model():
    return load_model("models/satellite_cnn.keras")


# -----------------------------
# Weather Prediction
# -----------------------------
def predict_weather():

    df = pd.read_csv(
        "data/weather/weather_prepared.csv"
    )

    features = [
        "temperature_2m_mean",
        "relative_humidity_2m_mean",
        "precipitation_sum",
        "pressure_msl_mean"
    ]

    data = df[features].values

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    sequence = scaled_data[-7:]
    X_input = np.array([sequence])

    model = load_weather_model()

    prediction_scaled = model.predict(
        X_input,
        verbose=0
    )

    prediction = scaler.inverse_transform(
        prediction_scaled
    )

    temperature = float(prediction[0][0])
    humidity = float(prediction[0][1])
    rainfall = max(0.0, float(prediction[0][2]))
    pressure = float(prediction[0][3])

    return temperature, humidity, rainfall, pressure


# -----------------------------
# Satellite Prediction
# -----------------------------
def predict_satellite(image):

    model = load_satellite_model()

    class_names = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake"
]

    img = load_img(
        image,
        target_size=(64, 64)
    )

    img_array = img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    prediction = model.predict(
        img_array,
        verbose=0
    )

    predicted_index = np.argmax(
        prediction[0]
    )

    predicted_class = class_names[
        predicted_index
    ]

    confidence = (
        prediction[0][predicted_index] * 100
    )

    return predicted_class, confidence


# -----------------------------
# Crop Recommendation
# -----------------------------
def recommend_crop(
    temperature,
    humidity,
    rainfall,
    land_type
):

    if land_type == "AnnualCrop":

        if rainfall >= 100 and humidity >= 60:
            return "Rice"

        elif 20 <= temperature <= 30:
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


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("🌦️ System Information")

st.sidebar.write(
    "Models used:"
)

st.sidebar.write(
    "• LSTM - Weather Prediction"
)

st.sidebar.write(
    "• CNN - Satellite Classification"
)

st.sidebar.write(
    "• Rule-Based Decision Engine - Crop Recommendation"
)


# -----------------------------
# Weather Section
# -----------------------------
st.header("🌦️ Next Day Weather Prediction")
st.divider()

if st.button("Predict Next Day Weather"):

    with st.spinner("Predicting weather..."):

        temperature, humidity, rainfall, pressure = (
            predict_weather()
        )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Temperature",
        f"{temperature:.2f} °C"
    )

    col2.metric(
        "Humidity",
        f"{humidity:.2f} %"
    )

    col3.metric(
        "Rainfall",
        f"{rainfall:.2f} mm"
    )

    col4.metric(
        "Pressure",
        f"{pressure:.2f} hPa"
    )

    st.session_state["temperature"] = temperature
    st.session_state["humidity"] = humidity
    st.session_state["rainfall"] = rainfall


# -----------------------------
# Satellite Image Section
# -----------------------------
st.header("🛰️ Satellite Image Analysis")
st.divider()

uploaded_image = st.file_uploader(
    "Upload a satellite image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_image is not None:

    st.image(
        uploaded_image,
        caption="Uploaded Satellite Image",
        width=400
    )

    if st.button("Classify Satellite Image"):

        with st.spinner("Analyzing satellite image..."):

            land_type, confidence = predict_satellite(
                uploaded_image
            )

        st.success(
            f"Land Type: {land_type}"
        )

        st.info(
            f"Classification Confidence: {confidence:.2f}%"
        )

        st.session_state["land_type"] = land_type


# -----------------------------
# Crop Recommendation Section
# -----------------------------
st.header("🌾 Smart Crop Recommendation")
st.divider()

if st.button("Generate Crop Recommendation"):

    if (
        "temperature" not in st.session_state
        or "humidity" not in st.session_state
        or "rainfall" not in st.session_state
        or "land_type" not in st.session_state
    ):

        st.warning(
            "Please predict the weather and classify a satellite image first."
        )

    else:

        crop = recommend_crop(
            st.session_state["temperature"],
            st.session_state["humidity"],
            st.session_state["rainfall"],
            st.session_state["land_type"]
        )

        st.success(
            f"🌱 Recommended Crop: {crop}"
        )