import joblib
import numpy as np
import pandas as pd

# Load model components
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

# Separate input features
numerical_features = [
    "longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms",
    "population", "households", "median_income"
]

ocean_categories = [
    "ocean_proximity_<1H_OCEAN",
    "ocean_proximity_INLAND",
    "ocean_proximity_ISLAND",
    "ocean_proximity_NEAR_BAY",
    "ocean_proximity_NEAR_OCEAN"
]

print("Enter the following housing information:")
input_values = []
for feature in numerical_features:
    val = float(input(f"{feature}: "))
    input_values.append(val)

# Ocean proximity (manual one-hot encoding)
print("\nOcean Proximity options: <1H_OCEAN, INLAND, ISLAND, NEAR_BAY, NEAR_OCEAN")
selected = input("Enter ocean proximity: ").strip()

for cat in ocean_categories:
    category_label = cat.split("_")[-1]
    input_values.append(1.0 if selected == category_label else 0.0)

# Create input DataFrame with full feature alignment
input_df = pd.DataFrame([input_values], columns=numerical_features + ocean_categories)

# Ensure all features match model's expectations
for col in feature_names:
    if col not in input_df.columns:
        input_df[col] = 0.0

# Reorder columns
input_df = input_df[feature_names]

# Scale and predict
input_scaled = scaler.transform(input_df)
prediction = model.predict(input_scaled)
print(f"\n🏡 Predicted Median House Value: ${prediction[0]:,.2f}")
