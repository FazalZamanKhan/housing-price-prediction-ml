
from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Load model, scaler, and feature names
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

def preprocess_input(data):
    # Convert input into DataFrame
    df = pd.DataFrame([data])

    # One-hot encode ocean_proximity manually
    ocean_cols = [col for col in feature_names if col.startswith("ocean_proximity_")]
    for col in ocean_cols:
        df[col] = 0
    ocean_col = f"ocean_proximity_{data['ocean_proximity']}"
    if ocean_col in df:
        df[ocean_col] = 1

    df.drop("ocean_proximity", axis=1, inplace=True)

    # Reorder columns to match training order
    df = df.reindex(columns=feature_names, fill_value=0)

    # Scale features
    scaled = scaler.transform(df)
    return scaled

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = {
            "longitude": float(request.form["longitude"]),
            "latitude": float(request.form["latitude"]),
            "housing_median_age": float(request.form["housing_median_age"]),
            "total_rooms": float(request.form["total_rooms"]),
            "total_bedrooms": float(request.form["total_bedrooms"]),
            "population": float(request.form["population"]),
            "households": float(request.form["households"]),
            "median_income": float(request.form["median_income"]),
            "ocean_proximity": request.form["ocean_proximity"]
        }

        X = preprocess_input(data)
        prediction = model.predict(X)[0]
        prediction = round(prediction, 2)

        return render_template("index.html", prediction_text=f"\U0001F3E1 Predicted Median House Value: ${prediction:,}")

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {e}")

if __name__ == "__main__":
    app.run(debug=True)
