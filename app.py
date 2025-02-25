from flask import Flask, request, jsonify
import pandas as pd
import joblib
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Load the trained model, label encoder, and scaler
try:
    model = joblib.load("crop_predictionmodel.pkl")
    label_encoder = joblib.load('label_encoder.pkl')
    scaler = joblib.load('scaler.pkl')
except Exception as e:
    raise RuntimeError(f"Error loading model, label encoder, or scaler: {str(e)}")

@app.route('/')
def home():
    return "Crop Prediction API is running."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        temperature = float(data['temperature'])
        humidity = float(data['humidity'])
        ph = float(data['ph'])
        nitrogen = float(data['nitrogen'])
        phosphorus = float(data['phosphorus'])
        potassium = float(data['potassium'])

        input_data = pd.DataFrame([{
            'Ph': ph,
            'N': nitrogen,
            'P': phosphorus,
            'K': potassium,
            'Temperature': temperature,
            'Humidity': humidity
        }])

        # Apply scaling before prediction
        input_scaled = scaler.transform(input_data)

        # Predict using the trained model
        prediction = model.predict(input_scaled)
        predicted_crop = label_encoder.inverse_transform(prediction)[0]

        return jsonify({"predicted_crop": predicted_crop})

    except KeyError as e:
        return jsonify({"error": f"Missing key: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)

