import joblib
import pandas as pd
import os

MODEL_DIR = "aqi_model_saved"
features = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))
print("Pickled features count:", len(features))
print("Pickled features list:", features)

df = pd.read_csv("Final_Dataset.csv")
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# Shift temporal leakage columns
for col in ['AQI_roll_mean_7', 'AQI_ewma_7', 'AQI_diff_1']:
    if col in df.columns:
        df[col] = df[col].shift(1)

# Feature engineering
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

df_cols = set(df.columns)
missing = [f for f in features if f not in df_cols]
print("Missing columns in engineered dataframe:", missing)
