import os
import numpy as np
from pathlib import Path

from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array

# Load trained CNN model
model = load_model("models/satellite_cnn.keras")

# Class names
class_names = sorted([
    folder.name
    for folder in Path("data/satellite").iterdir()
    if folder.is_dir()
])

# Find one sample satellite image
image_files = list(Path("data/satellite/AnnualCrop").glob("*.jpg"))

if not image_files:
    print("No image found!")
    exit()

image_path = image_files[0]

# Load and prepare image
img = load_img(image_path, target_size=(64, 64))
img_array = img_to_array(img)
img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Prediction
prediction = model.predict(img_array, verbose=0)

predicted_index = np.argmax(prediction[0])
predicted_class = class_names[predicted_index]
confidence = prediction[0][predicted_index] * 100

print("\nSatellite Image Prediction")
print("--------------------------")
print("Image:", image_path)
print("Predicted Class:", predicted_class)
print(f"Confidence: {confidence:.2f}%")