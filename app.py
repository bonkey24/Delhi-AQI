from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os

def asymmetric_loss(y_true, y_pred):
    residual = y_true - y_pred
    grad = np.where(residual > 0, -2.0 * residual, -residual)
    hess = np.where(residual > 0, 2.0, 1.0)
    return grad, hess

app = Flask(__name__)
CORS(app)

# Load the trained models and artifacts
MODEL_DIR = "aqi_model_saved"
try:
    stage1_regressor = joblib.load(os.path.join(MODEL_DIR, "stage1_regressor.pkl"))
    stage2_calibrated = joblib.load(os.path.join(MODEL_DIR, "stage2_calibrated.pkl"))
    thresholds = joblib.load(os.path.join(MODEL_DIR, "thresholds.pkl"))
    features = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))
    print("Models loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")

# Load the dataset to fetch feature values for the given date
try:
    df = pd.read_csv("Final_Dataset.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    # Fix Temporal Leakage
    for col in ['AQI_roll_mean_7', 'AQI_ewma_7', 'AQI_diff_1']:
        if col in df.columns:
            df[col] = df[col].shift(1)

    # Feature Engineering
    for i in [2, 3, 7, 14]:
        df[f'AQI_lag_{i}'] = df['AQI'].shift(i)

    for w in [3, 7, 14]:
        df[f'AQI_roll_mean_{w}'] = df['AQI'].rolling(w).mean().shift(1)
        df[f'AQI_roll_std_{w}']  = df['AQI'].rolling(w).std().shift(1)
        df[f'AQI_roll_max_{w}']  = df['AQI'].rolling(w).max().shift(1)

    df['ventilation']      = df['windspeed'] / (df['humidity'] + 1)
    df['ventilation_lag1'] = df['ventilation'].shift(1)
    df['windspeed_lag1']   = df['windspeed'].shift(1)
    df['humidity_lag1']    = df['humidity'].shift(1)
    df['temp_lag1']        = df['temp'].shift(1)

    df['AQI_delta_1']  = df['AQI'].diff().shift(1)
    df['is_spike']     = (df['AQI_delta_1'].abs() > 50).astype(int)

    # Ensure no dropna so we can predict for as many valid rows as possible, 
    # but we might just dropna or leave as is. We'll leave as is, since 
    # we just need the row for the requested date. Missing values will just be passed to the model.
    print("Dataset loaded and features engineered successfully.")
except Exception as e:
    print(f"Error loading dataset: {e}")

label_names = {0: "Good", 1: "Satisfactory", 2: "Moderate", 3: "Poor", 4: "Very Poor", 5: "Severe"}
category_colors = {
    "Good": "#4ade80",          # Green
    "Satisfactory": "#facc15",  # Yellow
    "Moderate": "#fb923c",      # Orange
    "Poor": "#f87171",          # Red
    "Very Poor": "#a855f7",     # Purple
    "Severe": "#881337"         # Dark Red
}

def apply_per_boundary_thresholds(probs, thresholds):
    preds = np.argmax(probs, axis=1)
    for (low, high), t in thresholds.items():
        mask = (preds == low) | (preds == high)
        if mask.sum() == 0:
            continue
        rel = probs[mask, high] / (probs[mask, low] + probs[mask, high] + 1e-9)
        preds[mask] = np.where(rel > t, high, low)
    return preds

@app.route('/predict', methods=['GET'])
def predict_aqi():
    date_str = request.args.get('date')
    if not date_str:
        return jsonify({'error': 'Please provide a date (YYYY-MM-DD)'}), 400
    
    try:
        target_date = pd.to_datetime(date_str)
    except ValueError:
        return jsonify({'error': 'Invalid date format. Use YYYY-MM-DD'}), 400

    row = df[df['Date'] == target_date]
    if row.empty:
        return jsonify({'error': f'No historical feature data found for {date_str}. Try a date between 2017 and 2025.'}), 404

    # Extract features for Stage 1
    X_input = row[features].copy()

    # Predict Stage 1 (Regression)
    pred_log = stage1_regressor.predict(X_input)
    pred_aqi = np.expm1(pred_log)[0]

    # Prepare features for Stage 2
    X_clf = X_input.copy()
    X_clf['Predicted_AQI'] = pred_aqi
    X_clf['alarm_predicted_poor'] = (pred_aqi >= 200).astype(int)
    X_clf['alarm_lag_poor'] = (X_clf['AQI_lag_1'] >= 200).astype(int)

    # Predict Stage 2 (Classification)
    probs = stage2_calibrated.predict_proba(X_clf)
    pred_cat_idx = apply_per_boundary_thresholds(probs, thresholds)[0]
    
    category = label_names[pred_cat_idx]
    
    # Grab previous 7 days of actual AQI for historical trend comparison
    prev_7_days = df[df['Date'] < target_date].tail(7)
    trend = []
    for _, r in prev_7_days.iterrows():
        trend.append({
            'date': r['Date'].strftime('%Y-%m-%d'),
            'aqi': float(r['AQI']),
            'category': label_names[get_cat(r['AQI'])] if 'get_cat' in globals() or 'get_cat' in locals() else "Unknown"
        })

    # Let's ensure get_cat is defined or fallback
    def local_get_cat(aqi):
        if aqi <= 50:    return "Good"
        elif aqi <= 100: return "Satisfactory"
        elif aqi <= 200: return "Moderate"
        elif aqi <= 300: return "Poor"
        elif aqi <= 400: return "Very Poor"
        else:            return "Severe"

    for t in trend:
        t['category'] = local_get_cat(t['aqi'])
        t['color'] = category_colors.get(t['category'], "#000000")
        
    return jsonify({
        'date': date_str,
        'predicted_aqi': round(pred_aqi, 2),
        'category': category,
        'color': category_colors.get(category, "#000000"),
        'actual_aqi': float(row['AQI'].values[0]) if 'AQI' in row else None,
        'weather': {
            'temp': float(row['temp'].values[0]) if 'temp' in row else None,
            'humidity': float(row['humidity'].values[0]) if 'humidity' in row else None,
            'precip': float(row['precip'].values[0]) if 'precip' in row else None,
            'windspeed': float(row['windspeed'].values[0]) if 'windspeed' in row else None,
            'sealevelpressure': float(row['sealevelpressure'].values[0]) if 'sealevelpressure' in row else None,
        },
        'trend': trend
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
