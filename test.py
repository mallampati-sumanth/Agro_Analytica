import joblib
import pandas as pd

# Load the model and label encoder
model = joblib.load("crop_predictionmodel.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Test with different inputs
test_data = pd.DataFrame([{
    'Ph': 6.5, 'N': 50, 'P': 30, 'K': 40, 'temperature': 25, 'humidity': 60
}])

# Predict
prediction = model.predict(test_data)
predicted_crop = label_encoder.inverse_transform(prediction)[0]
print(f"Predicted Crop: {predicted_crop}")
