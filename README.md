
# 🏠 California Housing Price Predictor

This project predicts **median house values** in California districts using advanced machine learning regression techniques. It supports both **terminal-based predictions** and a **Flask-based web interface**.

---

## 🚀 Overview

We trained and evaluated multiple models using:

- ✅ **Batch Gradient Descent** with L2 Regularization  
- ✅ **Stochastic Gradient Descent** with L1 Regularization  
- ✅ **Mini-Batch Gradient Descent** with ElasticNet Regularization  

The best-performing model was saved and used for real-time inference in both terminal and web-based formats.

---

## 📊 Dataset

- **File:** `housing.csv`  
- **Source:** California Housing Dataset (based on 1990 census)  
- **Target:** `median_house_value`  
- **Features Include:**  
  - `longitude`, `latitude`, `housing_median_age`, `total_rooms`, `total_bedrooms`  
  - `population`, `households`, `median_income`  
  - One-hot encoded: `ocean_proximity` (`INLAND`, `ISLAND`, `NEAR_BAY`, `NEAR_OCEAN`)  
    (Baseline `<1H_OCEAN` dropped)

---

## 🛠️ Model Training Details

- **Missing Values:** Handled using `SimpleImputer(strategy="median")`
- **Encoding:** One-Hot Encoding for `ocean_proximity` with `drop_first=True`
- **Scaling:** `StandardScaler` for numerical features
- **Algorithms Used:**  
  - `SGDRegressor` from `sklearn.linear_model`
  - Regularization: L1, L2, and ElasticNet
  - Early stopping enabled for better generalization

---

## 📈 Model Performance (on Test Set)

| Model                           | Mean Squared Error (MSE) |
|----------------------------------|---------------------------|
| Batch Gradient Descent (L2)      | 4.93B                     |
| **Stochastic GD (L1)**           | **4.88B ✅ Best**         |
| Mini-Batch GD (ElasticNet)       | 4.90B                     |

---

## 📦 Inference Options

### ▶️ 1. **Command-Line Prediction**

#### Run the script:
```bash
python inference.py
```

You will be prompted to enter values for each feature. The script will:
- Encode `ocean_proximity`
- Scale input using `StandardScaler`
- Output predicted house value

---

### 🌐 2. **Web App (Flask UI)**

#### Run the Flask app:
```bash
python app.py
```

Then open your browser at:

```
http://127.0.0.1:5000
```

You’ll see a form to enter input values and receive real-time predictions.

---

## 💻 File Structure

```
.
├── housing.csv             # Dataset
├── task.ipynb              # Jupyter Notebook with training
├── model.pkl               # Trained regression model
├── scaler.pkl              # Fitted StandardScaler used in training
├── feature_names.pkl       # List of columns used for prediction
├── inference.py            # CLI-based prediction script
├── app.py                  # Flask app backend
├── templates/
│   └── index.html          # Web UI template for prediction form
├── README.md               # Project documentation
```

---

## 🧩 Installation & Requirements

Install required packages:

```bash
pip install pandas numpy scikit-learn matplotlib flask joblib
```

---

## 🧪 Example Input (CLI or Web Form)

```
longitude: -118
latitude: 32
housing_median_age: 30
total_rooms: 4000
total_bedrooms: 800
population: 1800
households: 600
median_income: 3.5
ocean_proximity: INLAND
```

✅ Expected Output:

```
🏡 Predicted Median House Value: $197,394.34
```

---

## 📄 License

MIT License

---


Hugging Face repo link:
https://huggingface.co/fazal33/HOUSES_PRICE_PREDICTOR


web UI link:
http://127.0.0.1:5000


## 👤 Author

**Fazal Zaman**  
📧 Email: [fazalzamanper@gmail.com](mailto:fazalzamanper@gmail.com)

---
