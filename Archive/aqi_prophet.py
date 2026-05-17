
# ============================================================
#   Delhi AQI Prediction using Meta Prophet (Upgraded Model)
#   Data: 2020–2025 | City: Delhi, India
# ============================================================

import matplotlib
matplotlib.use("Agg")   # Non-interactive backend — saves plots without opening windows

import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# STEP 0: GENERATE REALISTIC DAILY AQI & WEATHER DATA
# ─────────────────────────────────────────────────────────────

print("=" * 60)
print("  DELHI AQI PREDICTION — Meta Prophet Model (V2.0)")
print("=" * 60)
print("\n[Step 0] Generating realistic daily AQI and weather data...")

yearly_data = {
    2020: 185, 2021: 209, 2022: 209, 2023: 204, 2024: 209, 2025: 200
}

monthly_mult = {
    1: 1.65, 2: 1.40, 3: 1.05, 4: 0.90, 5: 0.95, 6: 0.80,
    7: 0.55, 8: 0.50, 9: 0.70, 10: 1.10, 11: 1.75, 12: 1.55,
}

diwali_dates = ["2020-11-14", "2021-11-04", "2022-10-24", "2023-11-12", "2024-11-01"]

np.random.seed(42)
all_rows = []

for year, yearly_avg in yearly_data.items():
    dates = pd.date_range(f"{year}-01-01", f"{year}-05-15" if year == 2025 else f"{year}-12-31", freq="D")

    for date in dates:
        month = date.month
        base = yearly_avg * monthly_mult[month]
        weekday_effect = 1.05 if date.weekday() < 5 else 0.95

        # ── WEATHER GENERATION ──
        # Temperature (Delhi pattern: cold in Jan, hot in May/Jun)
        temp_mean = 25 - 10 * np.cos((date.dayofyear - 15) * 2 * np.pi / 365.25)
        temperature = np.random.normal(temp_mean, 2.0)
        
        # Wind speed (Delhi pattern: calm in winter, windy in summer)
        wind_mean = 8 + 4 * np.sin((date.dayofyear - 100) * 2 * np.pi / 365.25)
        wind_speed = max(1.0, np.random.normal(wind_mean, 1.5))
        
        # Stubble burning indicator (Oct 15 - Nov 15)
        is_stubble_burning = 1 if (month == 10 and date.day >= 15) or (month == 11 and date.day <= 15) else 0

        noise = np.random.normal(1.0, 0.12)
        aqi = base * weekday_effect * noise

        # Apply realistic physical effects to AQI
        aqi += (25 - temperature) * 1.5      # Cold traps pollution
        aqi -= (wind_speed - 8) * 2.5        # Wind blows pollution away
        if is_stubble_burning:
            aqi += np.random.uniform(40, 90) # Extra smoke

        for diwali in diwali_dates:
            d = pd.Timestamp(diwali)
            if abs((date - d).days) <= 1:
                aqi += np.random.uniform(150, 220)

        aqi = max(10, min(500, aqi))

        all_rows.append({
            "date": date, 
            "aqi": round(aqi, 1),
            "temperature": round(temperature, 1),
            "wind_speed": round(wind_speed, 1),
            "is_stubble_burning": is_stubble_burning
        })

raw_df = pd.DataFrame(all_rows)
raw_df.to_csv("Delhi_AQI_Daily.csv", index=False)
print(f"   ✓ Generated {len(raw_df)} daily records with Weather & Seasonality features")


# ─────────────────────────────────────────────────────────────
# STEP 1: LOAD DATA
# ─────────────────────────────────────────────────────────────
print("\n[Step 1] Loading data...")
df = pd.read_csv("Delhi_AQI_Daily.csv")
print(f"   Shape    : {df.shape}")
print(f"   Columns  : {df.columns.tolist()}")


# ─────────────────────────────────────────────────────────────
# STEP 2: PREPARE FOR PROPHET
# ─────────────────────────────────────────────────────────────
print("\n[Step 2] Preparing data for Prophet...")
df = df.rename(columns={"date": "ds", "aqi": "y"})
df["ds"] = pd.to_datetime(df["ds"])
df = df.sort_values("ds").reset_index(drop=True)
df["y"] = df["y"].interpolate(method="linear")
df = df.dropna(subset=["ds", "y"])


# ─────────────────────────────────────────────────────────────
# STEP 3: TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────
split_date = df["ds"].max() - pd.Timedelta(days=90)
train = df[df["ds"] <= split_date].copy()
test  = df[df["ds"] >  split_date].copy()


# ─────────────────────────────────────────────────────────────
# STEP 4: BUILD PROPHET MODEL
# ─────────────────────────────────────────────────────────────
print("\n[Step 3] Building advanced Prophet model...")
# Optimized hyperparameters
model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    seasonality_mode="multiplicative",
    changepoint_prior_scale=0.1,    # tuned value
    seasonality_prior_scale=10.0,   # tuned value
    interval_width=0.95,
)

model.add_country_holidays(country_name="IN")

# Add new regressors
model.add_regressor('temperature')
model.add_regressor('wind_speed')
model.add_regressor('is_stubble_burning')
print("   ✓ Regressors added: temperature, wind_speed, is_stubble_burning")

import logging
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

# ─────────────────────────────────────────────────────────────
# STEP 5: FIT THE MODEL
# ─────────────────────────────────────────────────────────────
print("\n[Step 4] Training Prophet model...")
model.fit(train)
print("   ✓ Model trained successfully!")


# ─────────────────────────────────────────────────────────────
# STEP 6: MAKE FUTURE PREDICTIONS
# ─────────────────────────────────────────────────────────────
print("\n[Step 5] Generating forecast (30 days into future)...")

future = model.make_future_dataframe(periods=30, freq="D")

# For the future dates, we need to supply expected weather.
# We will use the historical mean for that specific Day-of-Year.
historical_weather = df.groupby(df['ds'].dt.dayofyear)[['temperature', 'wind_speed']].mean()
future['dayofyear'] = future['ds'].dt.dayofyear
future = future.merge(historical_weather, left_on='dayofyear', right_index=True, how='left')

# Stubble burning logic for future
future['is_stubble_burning'] = future['ds'].apply(
    lambda x: 1 if (x.month == 10 and x.day >= 15) or (x.month == 11 and x.day <= 15) else 0
)

# For any missing day-of-year (e.g. leap years), fill with nearest.
future['temperature'] = future['temperature'].ffill().bfill()
future['wind_speed'] = future['wind_speed'].ffill().bfill()
future = future.drop(columns=['dayofyear'])

forecast = model.predict(future)

forecast["yhat"]       = forecast["yhat"].clip(lower=0)
forecast["yhat_lower"] = forecast["yhat_lower"].clip(lower=0)
forecast["yhat_upper"] = forecast["yhat_upper"].clip(upper=500)


# ─────────────────────────────────────────────────────────────
# STEP 7: EVALUATE ON TEST SET
# ─────────────────────────────────────────────────────────────
print("\n[Step 6] Evaluating model on test set...")

test_forecast = forecast[forecast["ds"].isin(test["ds"])][["ds", "yhat"]]
test_merged   = test.merge(test_forecast, on="ds")

mae  = mean_absolute_error(test_merged["y"], test_merged["yhat"])
rmse = mean_squared_error(test_merged["y"], test_merged["yhat"]) ** 0.5
r2   = r2_score(test_merged["y"], test_merged["yhat"])
mape = (abs((test_merged["y"] - test_merged["yhat"]) / test_merged["y"]).mean()) * 100

print(f"\n   📊 UPGRADED MODEL PERFORMANCE:")
print(f"   ┌──────────────────────────────────────┐")
print(f"   │  MAE  : {mae:>8.2f}  (AQI units)      │")
print(f"   │  RMSE : {rmse:>8.2f}                   │")
print(f"   │  R²   : {r2:>8.3f}  (1.0 = perfect)   │")
print(f"   │  MAPE : {mape:>7.2f}%                   │")
print(f"   └──────────────────────────────────────┘")


def aqi_category(aqi):
    if aqi <= 50:    return "Good"
    elif aqi <= 100: return "Satisfactory"
    elif aqi <= 200: return "Moderate"
    elif aqi <= 300: return "Poor"
    elif aqi <= 400: return "Very Poor"
    else:            return "Severe"

forecast["aqi_category"] = forecast["yhat"].apply(aqi_category)

# ─────────────────────────────────────────────────────────────
# STEP 8: VISUALIZATIONS
# ─────────────────────────────────────────────────────────────
print("\n[Step 7] Generating visualizations...")

try:
    plt.style.use("seaborn-v0_8-darkgrid")
except OSError:
    try:
        plt.style.use("seaborn-darkgrid")
    except OSError:
        plt.style.use("dark_background")
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.facecolor": "#0f1117",
    "axes.facecolor": "#1a1d27",
    "axes.labelcolor": "white",
    "axes.titlecolor": "white",
    "xtick.color": "white",
    "ytick.color": "white",
    "grid.color": "#2e3250",
    "text.color": "white",
    "legend.facecolor": "#1a1d27",
    "legend.edgecolor": "#444",
})

# ── Plot 1: Full Forecast
fig, ax = plt.subplots(figsize=(16, 7))
ax.fill_between(forecast["ds"], forecast["yhat_lower"], forecast["yhat_upper"], alpha=0.25, color="#4fc3f7", label="95% Confidence Interval")
ax.plot(df["ds"], df["y"], color="#78909c", linewidth=0.8, alpha=0.7, label="Historical AQI")
ax.plot(forecast["ds"], forecast["yhat"], color="#4fc3f7", linewidth=2, label="Predicted AQI")

future_start = df["ds"].max()
ax.axvspan(future_start, forecast["ds"].max(), alpha=0.1, color="#ff7043", label="Future Forecast")
ax.axvline(future_start, color="#ff7043", linestyle="--", linewidth=1.5, alpha=0.8)

for threshold, label, color in [(200, "Poor threshold", "#ff5252"), (300, "Very Poor", "#ce93d8")]:
    ax.axhline(threshold, color=color, linestyle=":", linewidth=1.2, alpha=0.6, label=label)

ax.set_title("Delhi AQI Forecast — Prophet Model (2020–2025 + 30-day Ahead)", fontsize=16, pad=15)
ax.set_xlabel("Date")
ax.set_ylabel("Air Quality Index (AQI)")
ax.legend(loc="upper left", fontsize=9)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("plot1_aqi_forecast.png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()

# ── Plot 2: Actual vs Predicted
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(test_merged["ds"], test_merged["y"], color="#4fc3f7", linewidth=2, label="Actual AQI", marker="o", markersize=3)
ax.plot(test_merged["ds"], test_merged["yhat"], color="#ff7043", linewidth=2, linestyle="--", label="Predicted AQI", marker="s", markersize=3)
ax.fill_between(test_merged["ds"], test_merged["y"], test_merged["yhat"], alpha=0.15, color="#ffb300", label="Prediction Error")

ax.set_title(f"Actual vs Predicted AQI — Test Set (Last 90 Days)\nMAE: {mae:.1f} | R²: {r2:.3f}", fontsize=13, pad=12)
ax.set_xlabel("Date")
ax.set_ylabel("AQI")
ax.legend(fontsize=10)
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("plot2_actual_vs_predicted.png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()

# ── Plot 3: Prophet Components
fig3 = model.plot_components(forecast)
fig3.set_facecolor("#0f1117")
for ax_ in fig3.get_axes():
    ax_.set_facecolor("#1a1d27")
    ax_.tick_params(colors="white")
    ax_.xaxis.label.set_color("white")
    ax_.yaxis.label.set_color("white")
    ax_.title.set_color("white")
    for line in ax_.get_lines(): line.set_color("#4fc3f7")
plt.suptitle("Prophet Components — Regressors & Seasonality", color="white", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig("plot3_components.png", dpi=150, bbox_inches="tight", facecolor="#0f1117")
plt.close()

print("   ✓ Plots generated and saved")

# ─────────────────────────────────────────────────────────────
# STEP 9: SAVE OUTPUTS
# ─────────────────────────────────────────────────────────────
print("\n[Step 8] Saving output files...")
forecast[["ds","yhat","yhat_lower","yhat_upper","trend","aqi_category"]].to_csv("aqi_predictions.csv", index=False)
print("   ✓ Saved: aqi_predictions.csv")

print("\n" + "=" * 60)
print("  ✅ UPGRADE PIPELINE COMPLETE!")
print("=" * 60)
