import requests
from requests.exceptions import RequestException
import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import os
from flask import Flask, request, jsonify, render_template
import reverse_geocoder as rg
# CRITICAL FIX: The template_folder needs to be set to '.' if index.html is in the same folder as app.py
# If index.html is in a 'templates' folder, change this back to: app = Flask(__name__)
app = Flask(__name__)

# ===============================================================
# 1️⃣ Reverse Geocode: Get State Name from Coordinates
# ===============================================================

def reverse_geocode_state(lat, lon):
    url = "https://nominatim.openstreetmap.org/reverse"
    params = {
        "format": "jsonv2",
        "lat": lat,
        "lon": lon,
        "zoom": 10,
        "addressdetails": 1
    }
    headers = {"User-Agent": "CropRecommender/1.0 (aryan.shinde@example.com)"}
    try:
        r = requests.get(url, params=params, headers=headers, timeout=15)
        r.raise_for_status()
        data = r.json()
    except RequestException as e:
        app.logger.error(f"Reverse geocode failed: {e}")
        return "Unknown"
    
    address = data.get("address", {})
    # Prioritize official administrative regions for state name
    for key in ("state", "region", "state_district", "province", "county"):
        if key in address:
            # Normalize to the title case used in STATE_CONDITIONS if possible
            return address[key].title() 
    return address.get("country", "Unknown").title()


# ===============================================================
# 2️⃣ Fetch Weather Forecast (Open-Meteo)
# ===============================================================

def fetch_open_meteo_forecast(lat, lon, days=7):
    if days < 1:
        raise ValueError("days must be >= 1")
    days = min(days, 16)
    base = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": ",".join([
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum"
        ]),
        "hourly": ",".join([
            "relativehumidity_2m",
            "soil_temperature_0cm",
            "soil_moisture_0_to_1cm",
            "temperature_2m"
        ]),
        "forecast_days": days,
        "timezone": "auto"
    }
    try:
        r = requests.get(base, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
    except RequestException as e:
        raise RuntimeError(f"Open-Meteo request failed: {e}")

    if "daily" not in data or "time" not in data["daily"]:
        raise RuntimeError("Open-Meteo returned unexpected data structure.")
        
    daily = data["daily"]
    hourly = data.get("hourly", {})
    dates = daily["time"]
    tmp_max = daily.get("temperature_2m_max", [None]*len(dates))
    tmp_min = daily.get("temperature_2m_min", [None]*len(dates))
    precip = daily.get("precipitation_sum", [0.0]*len(dates))
    
    hourly_time = hourly.get("time", [])
    rh = hourly.get("relativehumidity_2m", [])
    soil_temp = hourly.get("soil_temperature_0cm", [])
    soil_moist = hourly.get("soil_moisture_0_to_1cm", [])
    temp_hourly = hourly.get("temperature_2m", [])
    
    # Handle case where hourly data is missing
    if len(hourly_time) == 0:
        rows = []
        for i, d in enumerate(dates):
            rows.append({
                "date": pd.to_datetime(d).date(),
                "temp_max": tmp_max[i],
                "temp_min": tmp_min[i],
                "precipitation": precip[i],
                "rh_mean": None,
                "soil_temp_mean": None,
                "soil_moist_mean": None,
                "temp_mean": None
            })
        return pd.DataFrame(rows)

    # Process hourly data
    hourly_df = pd.DataFrame({
        "time": hourly_time,
        "relativehumidity_2m": rh,
        "soil_temperature_0cm": soil_temp,
        "soil_moisture_0_to_1cm": soil_moist,
        "temperature_2m": temp_hourly
    })
    hourly_df["time"] = pd.to_datetime(hourly_df["time"])
    hourly_df["date"] = hourly_df["time"].dt.date
    
    agg = hourly_df.groupby("date").agg({
        "relativehumidity_2m": "mean",
        "soil_temperature_0cm": "mean",
        "soil_moisture_0_to_1cm": "mean",
        "temperature_2m": "mean"
    }).reset_index()

    # Merge daily and aggregated hourly data
    daily_rows = []
    for i, dstr in enumerate(dates):
        ddate = pd.to_datetime(dstr).date()
        row = {
            "date": ddate,
            "temp_max": tmp_max[i],
            "temp_min": tmp_min[i],
            "precipitation": precip[i]
        }
        match = agg[agg["date"] == ddate]
        if not match.empty:
            row["rh_mean"] = float(match["relativehumidity_2m"].iloc[0])
            row["soil_temp_mean"] = float(match["soil_temperature_0cm"].iloc[0])
            row["soil_moist_mean"] = float(match["soil_moisture_0_to_1cm"].iloc[0])
            row["temp_mean"] = float(match["temperature_2m"].iloc[0])
        else:
            row["rh_mean"] = None
            row["soil_temp_mean"] = None
            row["soil_moist_mean"] = None
            row["temp_mean"] = None
        daily_rows.append(row)
        
    return pd.DataFrame(daily_rows)


# ===============================================================
# 3️⃣ Compute Aggregated Features for Model Input
# ===============================================================
def compute_features_from_forecast(df):
    for col in ["temp_max", "temp_min", "precipitation", "rh_mean", "soil_temp_mean", "soil_moist_mean", "temp_mean"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Temperature calculation
    if "temp_mean" in df.columns and df["temp_mean"].notnull().any():
        temp_avg = df["temp_mean"].mean(skipna=True)
    else:
        # Fallback to mean of daily max/min average
        temp_avg = ((df["temp_max"] + df["temp_min"]) / 2.0).mean(skipna=True)
        
    # Other averages
    humidity_avg = df["rh_mean"].mean(skipna=True) if "rh_mean" in df.columns else 0.0
    rainfall_avg = df["precipitation"].sum(skipna=True) / 7.0 if "precipitation" in df.columns else 0.0 # Adjusted to average daily rainfall over 7 days
    soil_moist_avg = df["soil_moist_mean"].mean(skipna=True) if "soil_moist_mean" in df.columns else 0.15
    soil_temp_avg = df["soil_temp_mean"].mean(skipna=True) if "soil_temp_mean" in df.columns else temp_avg
    
    # Handle NaN/None soil moisture
    if soil_moist_avg is None or pd.isna(soil_moist_avg) or soil_moist_avg == 0:
        soil_moist_avg = 0.15
        
    # Estimate NPK and pH (based on soil moisture/temp correlation)
    N = round(max(0.0, soil_moist_avg * 200.0), 2)
    P = round(max(0.0, soil_temp_avg * 4.0), 2)
    K = round(max(0.0, soil_moist_avg * 250.0), 2)
    ph = round(6.5 + (soil_temp_avg - 25.0) * 0.02, 2)

    features = {
        "N": N,
        "P": P,
        "K": K,
        "temperature": round(float(temp_avg) if not pd.isna(temp_avg) else 25.0, 2),
        "humidity": round(float(humidity_avg), 2),
        "ph": ph,
        "rainfall": round(float(rainfall_avg), 2)
    }
    return features


# ===============================================================
# 4️⃣ Season Code (based on month)
# ===============================================================

def get_season_code_from_month(month_name):
    month_name = month_name.lower()
    if month_name in ["october", "november", "december", "january", "february", "march"]:
        return 1  # Rabi
    elif month_name in ["june", "july", "august", "september"]:
        return 2  # Kharif
    elif month_name in ["april", "may"]:
        return 3  # Zaid
    else:
        return None

# ===============================================================
# 5️⃣ Model Loading & Prediction (WITH UNSEEN STATE FIX)
# ===============================================================

def load_model_and_predict(features):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, "xgboost_model.pkl")
    STATE_LE_PATH = os.path.join(BASE_DIR, "state_encoder.pkl")
    CROP_LE_PATH = os.path.join(BASE_DIR, "crop_encoder.pkl")

    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(STATE_LE_PATH, "rb") as f:
            state_le = pickle.load(f)
        with open(CROP_LE_PATH, "rb") as f:
            crop_le = pickle.load(f)

        state_name = features["state_code"][0].replace(" State", "").replace(" District", "").strip()

        # 👇 CRITICAL FIX: Gracefully handle unseen state names by using a default encoded value
        if state_name in state_le.classes_:
            # transform expects array-like input
            enc_state = state_le.transform(features["state_code"])[0] 
        else:
            app.logger.warning(f"State '{state_name}' not in encoder vocabulary. Using default state code 0.")
            enc_state = 0 # Use a default safe value

        X_input = np.array([[
            features["N"],
            features["P"],
            features["K"],
            features["temperature"],
            features["humidity"],
            features["ph"],
            features["rainfall"],
            enc_state,
            features["season_code"]
        ]])

        prediction = model.predict(X_input)
        prediction_crop = crop_le.inverse_transform(prediction)[0]
        return prediction_crop.capitalize()


    except FileNotFoundError:
        app.logger.error("Model or encoder files not found.")
        return "FileERROR"
    except Exception as e:
        app.logger.error(f"Prediction/Encoding error: {e}")
        return "ModelERROR"


# ===============================================================
# 🌾 Alternative Crops Logic (Requirement Data)
# ===============================================================

# Average Daily Rainfall Version (mm/day)
CROP_REQUIREMENTS = {
    # Cereals
    "rice": {"temperature": (20, 35), "humidity": (70, 90), "rainfall": (1.25, 2.5), "ph": (5.5, 7.0)},
    "maize": {"temperature": (18, 30), "humidity": (50, 80), "rainfall": (0.67, 1.25), "ph": (5.8, 7.0)},
    "wheat": {"temperature": (10, 25), "humidity": (50, 70), "rainfall": (0.42, 0.83), "ph": (6.0, 7.5)},
    "barley": {"temperature": (12, 25), "humidity": (40, 60), "rainfall": (0.33, 0.83), "ph": (6.0, 7.5)},
    "millet": {"temperature": (25, 35), "humidity": (40, 60), "rainfall": (0.25, 0.83), "ph": (5.5, 7.0)},
    "sorghum": {"temperature": (25, 35), "humidity": (40, 60), "rainfall": (0.33, 1.0), "ph": (6.0, 7.5)},

    # Pulses
    "chickpea": {"temperature": (10, 30), "humidity": (40, 60), "rainfall": (0.42, 0.83), "ph": (6.0, 8.0)},
    "kidneybeans": {"temperature": (15, 30), "humidity": (50, 70), "rainfall": (0.5, 1.0), "ph": (6.0, 7.5)},
    "blackgram": {"temperature": (20, 35), "humidity": (50, 70), "rainfall": (0.42, 0.83), "ph": (6.0, 7.5)},
    "lentil": {"temperature": (10, 30), "humidity": (40, 60), "rainfall": (0.33, 0.67), "ph": (6.0, 7.5)},
    "mungbean": {"temperature": (25, 35), "humidity": (50, 70), "rainfall": (0.42, 0.83), "ph": (6.2, 7.2)},
    "mothbeans": {"temperature": (25, 40), "humidity": (20, 50), "rainfall": (0.17, 0.5), "ph": (6.0, 7.0)},
    "pigeonpeas": {"temperature": (20, 35), "humidity": (50, 70), "rainfall": (0.42, 1.0), "ph": (6.0, 7.5)},

    # Commercial
    "cotton": {"temperature": (25, 35), "humidity": (50, 80), "rainfall": (0.42, 1.25), "ph": (6.0, 8.0)},
    "jute": {"temperature": (20, 35), "humidity": (70, 90), "rainfall": (1.25, 2.08), "ph": (6.4, 7.2)},
    "sugarcane": {"temperature": (20, 35), "humidity": (70, 85), "rainfall": (0.83, 2.08), "ph": (6.0, 7.5)},
    "coffee": {"temperature": (20, 30), "humidity": (60, 90), "rainfall": (1.25, 2.08), "ph": (6.0, 6.8)},
    "tea": {"temperature": (18, 30), "humidity": (70, 90), "rainfall": (1.25, 2.5), "ph": (4.5, 6.0)},
    "rubber": {"temperature": (25, 35), "humidity": (70, 90), "rainfall": (1.25, 2.5), "ph": (4.5, 6.5)},
    "tobacco": {"temperature": (20, 30), "humidity": (50, 70), "rainfall": (0.42, 0.83), "ph": (5.5, 6.5)},
    "groundnut": {"temperature": (25, 35), "humidity": (50, 70), "rainfall": (0.42, 1.0), "ph": (6.0, 7.0)},
    "sunflower": {"temperature": (20, 30), "humidity": (50, 70), "rainfall": (0.42, 0.83), "ph": (6.0, 7.5)},
    "soybean": {"temperature": (20, 30), "humidity": (60, 80), "rainfall": (0.5, 1.0), "ph": (6.0, 7.5)},
    "mustard": {"temperature": (10, 25), "humidity": (40, 60), "rainfall": (0.25, 0.83), "ph": (6.0, 7.5)},

    # Fruits
    "banana": {"temperature": (25, 30), "humidity": (70, 90), "rainfall": (0.83, 1.67), "ph": (6.0, 7.5)},
    "mango": {"temperature": (24, 35), "humidity": (50, 70), "rainfall": (0.42, 1.25), "ph": (5.5, 7.5)},
    "orange": {"temperature": (20, 30), "humidity": (50, 70), "rainfall": (0.5, 1.0), "ph": (5.5, 7.0)},
    "grapes": {"temperature": (20, 30), "humidity": (50, 70), "rainfall": (0.33, 0.83), "ph": (6.0, 7.5)},
    "papaya": {"temperature": (25, 35), "humidity": (60, 80), "rainfall": (0.67, 1.25), "ph": (6.0, 6.5)},
    "pomegranate": {"temperature": (25, 35), "humidity": (50, 70), "rainfall": (0.42, 0.83), "ph": (6.0, 7.5)},
    "guava": {"temperature": (23, 30), "humidity": (50, 70), "rainfall": (0.5, 0.83), "ph": (6.0, 7.5)},
    "apple": {"temperature": (10, 25), "humidity": (50, 70), "rainfall": (0.42, 1.25), "ph": (6.0, 7.5)},
    "pineapple": {"temperature": (22, 32), "humidity": (70, 90), "rainfall": (1.25, 2.08), "ph": (4.5, 6.5)},
    "watermelon": {"temperature": (25, 35), "humidity": (50, 70), "rainfall": (0.42, 0.83), "ph": (6.0, 7.5)},
    "muskmelon": {"temperature": (24, 32), "humidity": (50, 70), "rainfall": (0.42, 0.83), "ph": (6.0, 7.5)},

    # Plantation
    "coconut": {"temperature": (25, 35), "humidity": (70, 90), "rainfall": (1.25, 2.08), "ph": (5.5, 7.0)},
    "cashew": {"temperature": (24, 35), "humidity": (50, 70), "rainfall": (0.42, 1.67), "ph": (5.0, 7.0)},
}

# --- State Average Environmental Conditions (Daily) ---
STATE_CONDITIONS = {
    "Andhra Pradesh": {"temperature": 28.5, "humidity": 75, "ph": 6.8, "rainfall": 5.2},
    "Arunachal Pradesh": {"temperature": 22.0, "humidity": 85, "ph": 6.2, "rainfall": 6.5},
    "Assam": {"temperature": 26.5, "humidity": 88, "ph": 6.0, "rainfall": 7.0},
    "Bihar": {"temperature": 27.0, "humidity": 70, "ph": 6.5, "rainfall": 4.5},
    "Chhattisgarh": {"temperature": 27.5, "humidity": 75, "ph": 6.6, "rainfall": 5.0},
    "Goa": {"temperature": 29.0, "humidity": 85, "ph": 6.5, "rainfall": 6.2},
    "Gujarat": {"temperature": 30.0, "humidity": 60, "ph": 7.0, "rainfall": 3.5},
    "Haryana": {"temperature": 26.0, "humidity": 55, "ph": 7.2, "rainfall": 2.8},
    "Himachal Pradesh": {"temperature": 18.0, "humidity": 65, "ph": 6.8, "rainfall": 4.0},
    "Jharkhand": {"temperature": 26.5, "humidity": 70, "ph": 6.4, "rainfall": 4.6},
    "Karnataka": {"temperature": 27.5, "humidity": 80, "ph": 6.4, "rainfall": 4.0},
    "Kerala": {"temperature": 28.0, "humidity": 88, "ph": 6.2, "rainfall": 7.5},
    "Madhya Pradesh": {"temperature": 27.0, "humidity": 65, "ph": 6.7, "rainfall": 3.8},
    "Maharashtra": {"temperature": 28.0, "humidity": 70, "ph": 6.6, "rainfall": 4.1},
    "Manipur": {"temperature": 23.0, "humidity": 80, "ph": 6.1, "rainfall": 6.2},
    "Meghalaya": {"temperature": 22.0, "humidity": 90, "ph": 5.8, "rainfall": 8.0},
    "Mizoram": {"temperature": 23.5, "humidity": 85, "ph": 6.0, "rainfall": 6.8},
    "Nagaland": {"temperature": 24.0, "humidity": 80, "ph": 6.1, "rainfall": 6.0},
    "Odisha": {"temperature": 28.0, "humidity": 80, "ph": 6.5, "rainfall": 5.5},
    "Punjab": {"temperature": 26.5, "humidity": 60, "ph": 7.3, "rainfall": 3.0},
    "Rajasthan": {"temperature": 31.0, "humidity": 45, "ph": 7.5, "rainfall": 2.0},
    "Sikkim": {"temperature": 20.0, "humidity": 85, "ph": 6.0, "rainfall": 6.5},
    "Tamil Nadu": {"temperature": 29.0, "humidity": 75, "ph": 6.7, "rainfall": 4.0},
    "Telangana": {"temperature": 28.0, "humidity": 70, "ph": 6.5, "rainfall": 4.2},
    "Tripura": {"temperature": 25.5, "humidity": 85, "ph": 6.3, "rainfall": 6.0},
    "Uttar Pradesh": {"temperature": 27.0, "humidity": 65, "ph": 7.0, "rainfall": 3.5},
    "Uttarakhand": {"temperature": 21.0, "humidity": 70, "ph": 6.8, "rainfall": 4.2},
    "West Bengal": {"temperature": 27.5, "humidity": 80, "ph": 6.3, "rainfall": 5.0},
    # Union Territories
    "Andaman and Nicobar Islands": {"temperature": 27.0, "humidity": 85, "ph": 6.5, "rainfall": 7.2},
    "Chandigarh": {"temperature": 26.0, "humidity": 60, "ph": 7.2, "rainfall": 3.0},
    "Dadra and Nagar Haveli and Daman and Diu": {"temperature": 28.0, "humidity": 75, "ph": 6.8, "rainfall": 5.0},
    "Delhi": {"temperature": 27.0, "humidity": 55, "ph": 7.3, "rainfall": 2.5},
    "Jammu and Kashmir": {"temperature": 16.0, "humidity": 65, "ph": 6.8, "rainfall": 3.8},
    "Ladakh": {"temperature": 10.0, "humidity": 40, "ph": 7.0, "rainfall": 1.5},
    "Lakshadweep": {"temperature": 28.0, "humidity": 85, "ph": 6.5, "rainfall": 7.0},
    "Puducherry": {"temperature": 29.0, "humidity": 80, "ph": 6.8, "rainfall": 4.8}
}

def recommend_alternative_crops(predicted_crop, state, top_n=10):
    env = STATE_CONDITIONS.get(state)
    if not env:
        return []

    crop_scores = []
    
    # Use the rainfall feature from the environment (STATE_CONDITIONS)
    env_rainfall_daily = env["rainfall"]

    for crop, req in CROP_REQUIREMENTS.items():
        if crop != predicted_crop:
            ideal_temp = sum(req["temperature"]) / 2
            ideal_hum = sum(req["humidity"]) / 2
            ideal_ph = sum(req["ph"]) / 2
            
            # Use the daily average rainfall requirement
            ideal_rain_daily = sum(req["rainfall"]) / 7 / 2 # Example scaling

            # Calculate score based on deviation from ideal
            score = (
                abs(env["temperature"] - ideal_temp)
                + abs(env["humidity"] - ideal_hum) / 2
                + abs(env["ph"] - ideal_ph) * 5
                + abs(env_rainfall_daily - ideal_rain_daily) * 2
            )
            crop_scores.append((crop, score))

    crop_scores.sort(key=lambda x: x[1])
    top_crops = [crop for crop, _ in crop_scores[:top_n]]
    return top_crops


# ===============================================================
# 🔹 Flask Routes
# ===============================================================

@app.route("/")
def home():
    # serve the provided index.html
    return render_template("index.html")



@app.route("/recommend", methods=["POST"])
def recommend_crop():
    data = request.get_json(force=True)
    auto_location = data.get("auto_location", False)

    try:
        # 1️⃣ GET COORDINATES
        lat = data.get("latitude")
        lon = data.get("longitude")
        
        if lat is None or lon is None:
            raise Exception("Latitude and Longitude must be provided.")
        
        lat = float(lat)
        lon = float(lon)

        # 2️⃣ FETCH WEATHER + SOIL DATA
        weather_and_soil_df = fetch_open_meteo_forecast(lat, lon, days=7)

        # 3️⃣ COMPUTE FEATURES
        features = compute_features_from_forecast(weather_and_soil_df)

        # 4️⃣ ADD SEASON + STATE CODES
        current_month = datetime.now().strftime("%B")
        season_code = get_season_code_from_month(current_month)
        state_name = reverse_geocode_state(lat, lon) # e.g., "Maharashtra"

        features["season_code"] = season_code
        features["state_code"] = [state_name] # state encoder expects array-like input

        # 5️⃣ MODEL PREDICTION
        predicted_crop = load_model_and_predict(features)

        # 6️⃣ RECOMMEND ALTERNATIVES
        alternatives = recommend_alternative_crops(predicted_crop, state_name, top_n=5)

        # 7️⃣ Compose response
        predicted_score = 1.0 # Placeholder score

        response_data = {
            "status": "success",
            "coords": {"latitude": lat, "longitude": lon},
            "weather": features, # Contains N, P, K, T, H, pH, R
          "predicted_crop": predicted_crop if isinstance(predicted_crop, str) else str(predicted_crop),

            "predicted_score": predicted_score,
            "fully_suitable": [],
            "partially_suitable": alternatives,
            "state": state_name,
            "season_code": season_code
        }
        return jsonify(response_data)

    except Exception as e:
        app.logger.error(f"Application ERROR: {e}")
        return jsonify({"status": "error", "message": str(e)}), 400


# ===============================================================
# Run Flask App
# ===============================================================

if __name__ == "__main__":
    # Ensure you run this inside an environment with the model/encoder files
    app.run(debug=True)