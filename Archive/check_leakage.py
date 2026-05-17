import pandas as pd

df = pd.read_csv('Final_Dataset.csv')

print("=== CHECKING FOR DATA LEAKAGE ===")
print("If the feature is calculated using the 'AQI' from the SAME DAY, that is temporal leakage.\n")

for i in range(15, 20):
    current_date = df['Date'].iloc[i]
    current_aqi = df['AQI'].iloc[i]
    
    # 1. Check AQI_roll_mean_7
    actual_roll = df['AQI_roll_mean_7'].iloc[i]
    leakage_roll = df['AQI'].iloc[i-6:i+1].mean()  # Includes today (Leakage!)
    safe_roll = df['AQI'].iloc[i-7:i].mean()       # Excludes today (Safe)
    
    # 2. Check AQI_ewma_7
    actual_ewma = df['AQI_ewma_7'].iloc[i]
    # To check EWMA roughly, we can use pandas ewm
    leakage_ewma = df['AQI'].iloc[:i+1].ewm(span=7, adjust=False).mean().iloc[-1]
    safe_ewma = df['AQI'].iloc[:i].ewm(span=7, adjust=False).mean().iloc[-1]

    print(f"Date: {current_date} | Today's Actual AQI: {current_aqi}")
    print(f"  [Roll Mean 7] In Dataset: {actual_roll:.2f}")
    print(f"      Calculated including today (Leakage) : {leakage_roll:.2f}")
    print(f"      Calculated excluding today (Safe)    : {safe_roll:.2f}")
    
    print(f"  [EWMA 7] In Dataset: {actual_ewma:.2f}")
    print(f"      Calculated including today (Leakage) : {leakage_ewma:.2f}")
    print(f"      Calculated excluding today (Safe)    : {safe_ewma:.2f}\n")

print("=== WEATHER FEATURE IMPORTANCE ===")
import xgboost as xgb
# Load the model we trained (we'll just quickly retrain it on a subset to get the exact numbers)
features = [col for col in df.columns if col not in ['Date', 'index', 'AQI']]
df = df.dropna().reset_index(drop=True)
model = xgb.XGBRegressor(n_estimators=50, max_depth=3, random_state=42, n_jobs=-1)
model.fit(df[features], df['AQI'])

imp = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_})
weather_features = ['temp', 'humidity', 'precip', 'windspeed', 'sealevelpressure']
weather_imp = imp[imp['Feature'].isin(weather_features)]
print(weather_imp.to_string(index=False))
