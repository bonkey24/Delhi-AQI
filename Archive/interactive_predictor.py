import pandas as pd
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

def train_model():
    print("Loading historical data and training ADVANCED Prophet model... (takes ~5 seconds)")
    try:
        df = pd.read_csv("Delhi_AQI_Daily.csv")
    except FileNotFoundError:
        print("Error: 'Delhi_AQI_Daily.csv' not found. Make sure you ran the main pipeline first.")
        return None, None
        
    df = df.rename(columns={"date": "ds", "aqi": "y"})
    df['ds'] = pd.to_datetime(df['ds'])
    
    # Pre-calculate historical daily averages for weather
    df['dayofyear'] = df['ds'].dt.dayofyear
    historical_weather = df.groupby('dayofyear')[['temperature', 'wind_speed']].mean()
    
    # Train advanced model
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        changepoint_prior_scale=0.1,
        seasonality_prior_scale=10.0,
        interval_width=0.95,
    )
    model.add_country_holidays(country_name="IN")
    
    # Adding regressors
    model.add_regressor('temperature')
    model.add_regressor('wind_speed')
    model.add_regressor('is_stubble_burning')
    
    # Supress Prophet's training logs
    import logging
    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
    
    model.fit(df)
    return model, historical_weather

def aqi_category(aqi):
    if aqi <= 50:    return "Good"
    elif aqi <= 100: return "Satisfactory"
    elif aqi <= 200: return "Moderate"
    elif aqi <= 300: return "Poor"
    elif aqi <= 400: return "Very Poor"
    else:            return "Severe"

def main():
    model, historical_weather = train_model()
    if model is None:
        return
        
    print("\n" + "="*60)
    print("  DELHI AQI INTERACTIVE PREDICTOR (V2 - Weather Aware)")
    print("="*60)
    print("You can predict the AQI for any past or future date!")
    
    while True:
        user_input = input("\nEnter a date (YYYY-MM-DD) or 'q' to quit: ").strip()
        
        if user_input.lower() in ['q', 'quit', 'exit']:
            print("Exiting Predictor. Goodbye!")
            break
            
        try:
            target_date = pd.to_datetime(user_input)
        except Exception:
            print("⚠️ Invalid date format. Please use YYYY-MM-DD (e.g., 2025-10-15).")
            continue
            
        # Create dataframe for prediction
        future = pd.DataFrame({'ds': [target_date]})
        
        # Look up expected weather based on historical daily averages
        doy = target_date.dayofyear
        # Handle leap years or edge cases by mapping to nearest valid DOY
        if doy not in historical_weather.index:
            doy = 365
            
        future['temperature'] = historical_weather.loc[doy, 'temperature']
        future['wind_speed'] = historical_weather.loc[doy, 'wind_speed']
        future['is_stubble_burning'] = 1 if (target_date.month == 10 and target_date.day >= 15) or (target_date.month == 11 and target_date.day <= 15) else 0
        
        forecast = model.predict(future)
        
        # Clip values to realistic bounds (0 - 500)
        yhat = max(0, min(500, forecast['yhat'].iloc[0]))
        yhat_lower = max(0, min(500, forecast['yhat_lower'].iloc[0]))
        yhat_upper = max(0, min(500, forecast['yhat_upper'].iloc[0]))
        
        cat = aqi_category(yhat)
        
        print("\n   +----------------------------------------------------+")
        print(f"   |  Date:     {target_date.strftime('%A, %d %B %Y')}")
        print(f"   |  AQI:      {yhat:.1f}")
        print(f"   |  Category: {cat}")
        print(f"   |  Range:    {yhat_lower:.1f} - {yhat_upper:.1f} (95% Confidence)")
        print(f"   |  Context:  Est. Temp: {future['temperature'].iloc[0]:.1f}°C | Est. Wind: {future['wind_speed'].iloc[0]:.1f}km/h")
        if future['is_stubble_burning'].iloc[0] == 1:
            print(f"   |            ⚠️ ACTIVE STUBBLE BURNING SEASON")
        print("   +----------------------------------------------------+")

if __name__ == "__main__":
    main()
