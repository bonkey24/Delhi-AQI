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

# CRITICAL FIX: Inject function into __main__ so joblib can unpickle the model under gunicorn
import sys
sys.modules['__main__'].asymmetric_loss = asymmetric_loss

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

@app.route('/')
def home():
    return jsonify({
        'status': 'online',
        'project': 'Delhi Air Quality Index Predictor API',
        'architecture': 'Two-Stage Machine Learning Ensemble (LightGBM + XGBoost + Ridge)',
        'usage': 'Query the model using /predict?date=YYYY-MM-DD (e.g. /predict?date=2024-05-17)'
    })

def safe_float(val, default=0.0):
    if pd.isna(val) or val is None:
        return default
    try:
        f = float(val)
        if np.isnan(f) or np.isinf(f):
            return default
        return f
    except (ValueError, TypeError):
        return default

@app.route('/predict', methods=['GET'])
def predict_aqi():
    import traceback
    try:
        date_str = request.args.get('date')
        if not date_str:
            return jsonify({'error': 'Please provide a date (YYYY-MM-DD)'}), 400
        
        try:
            target_date = pd.to_datetime(date_str)
        except ValueError:
            return jsonify({'error': 'Invalid date format. Use YYYY-MM-DD'}), 400

        is_future = False
        row = df[df['Date'] == target_date]
        
        if row.empty:
            is_future = True
            target_month = target_date.month
            target_day = target_date.day
            
            # Find matching historical dates (same month and day)
            historical_rows = df[(df['Date'].dt.month == target_month) & (df['Date'].dt.day == target_day)]
            if historical_rows.empty:
                historical_rows = df[df['Date'].dt.month == target_month]
            if historical_rows.empty:
                historical_rows = df
                
            # Synthesize features based on average historical values for this specific day of the year
            X_input = historical_rows[features].mean().to_frame().T
            # Safeguard: fill any NaN features with global means to prevent model crashes or NaN outputs
            X_input = X_input.fillna(df[features].mean()).fillna(0.0)
            
            weather_data = {
                'temp': safe_float(historical_rows['temp'].mean(), 25.0) if 'temp' in df else 25.0,
                'humidity': safe_float(historical_rows['humidity'].mean(), 50.0) if 'humidity' in df else 50.0,
                'precip': safe_float(historical_rows['precip'].mean(), 0.0) if 'precip' in df else 0.0,
                'windspeed': safe_float(historical_rows['windspeed'].mean(), 10.0) if 'windspeed' in df else 10.0,
                'sealevelpressure': safe_float(historical_rows['sealevelpressure'].mean(), 1010.0) if 'sealevelpressure' in df else 1010.0,
            }
            actual_aqi = None
            
            # Generate trend context based on historical preceding days
            trend = []
            for offset in range(7, 0, -1):
                prev_date = target_date - pd.Timedelta(days=offset)
                p_month = prev_date.month
                p_day = prev_date.day
                p_hist = df[(df['Date'].dt.month == p_month) & (df['Date'].dt.day == p_day)]
                if p_hist.empty:
                    p_hist = df[df['Date'].dt.month == p_month]
                
                p_aqi = safe_float(p_hist['AQI'].mean(), 150.0) if not p_hist.empty else 150.0
                trend.append({
                    'date': prev_date.strftime('%Y-%m-%d'),
                    'aqi': round(p_aqi, 2)
                })
        else:
            # Extract features for Stage 1
            X_input = row[features].copy()
            X_input = X_input.fillna(df[features].mean()).fillna(0.0)
            
            weather_data = {
                'temp': safe_float(row['temp'].values[0], 25.0) if 'temp' in row else 25.0,
                'humidity': safe_float(row['humidity'].values[0], 50.0) if 'humidity' in row else 50.0,
                'precip': safe_float(row['precip'].values[0], 0.0) if 'precip' in row else 0.0,
                'windspeed': safe_float(row['windspeed'].values[0], 10.0) if 'windspeed' in row else 10.0,
                'sealevelpressure': safe_float(row['sealevelpressure'].values[0], 1010.0) if 'sealevelpressure' in row else 1010.0,
            }
            actual_aqi = safe_float(row['AQI'].values[0], None)
            
            # Grab previous 7 days of actual AQI for historical trend comparison
            prev_7_days = df[df['Date'] < target_date].tail(7)
            trend = []
            for _, r in prev_7_days.iterrows():
                trend.append({
                    'date': r['Date'].strftime('%Y-%m-%d'),
                    'aqi': safe_float(r['AQI'], 150.0)
                })

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
            'actual_aqi': actual_aqi,
            'weather': weather_data,
            'trend': trend,
            'is_future': is_future,
            'prediction_type': 'Seasonal Climatology Synthesis' if is_future else 'Historical Baseline'
        })
    except Exception as e:
        err_tb = traceback.format_exc()
        print("CRITICAL SERVER ERROR:")
        print(err_tb)
        return jsonify({
            'error': str(e),
            'traceback': err_tb
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
