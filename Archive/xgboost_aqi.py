import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib
matplotlib.use("Agg")

print("==================================================")
print("  XGBOOST AQI PREDICTION (REAL DATA)")
print("==================================================")

# 1. Load Data
print("\n[Step 1] Loading Real Dataset...")
df = pd.read_csv("Final_Dataset.csv")
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# 2. Prepare Features and Target
print("\n[Step 2] Preparing Features...")
target = 'AQI'
features = [col for col in df.columns if col not in ['Date', 'index', 'AQI']]

print(f"   Features used ({len(features)}): {', '.join(features[:5])}... etc.")

# Drop rows with NaN (due to lagging/rolling means)
df = df.dropna().reset_index(drop=True)

# 3. Train / Test Split
# We will use the last 180 days as the test set to evaluate real-world forecasting
print("\n[Step 3] Chronological Train/Test Split...")
split_date = df['Date'].max() - pd.Timedelta(days=180)

train = df[df['Date'] <= split_date].copy()
test  = df[df['Date'] > split_date].copy()

X_train, y_train = train[features], train[target]
X_test, y_test   = test[features], test[target]

print(f"   Train set: {len(X_train)} days ({train['Date'].min().date()} to {train['Date'].max().date()})")
print(f"   Test set:  {len(X_test)} days ({test['Date'].min().date()} to {test['Date'].max().date()})")

# 4. Train XGBoost Model
print("\n[Step 4] Training XGBoost Regressor (Tuned to Prevent Overfitting)...")
model = xgb.XGBRegressor(
    n_estimators=250,
    learning_rate=0.04,
    max_depth=3,               # Reduced from 6 to stop trees from becoming too complex
    min_child_weight=15,       # Forces leaves to have more samples (stops memorization)
    subsample=0.6,             # Train on only 60% of data per tree
    colsample_bytree=0.7,      # Use only 70% of features per tree
    gamma=1.0,                 # Prunes trees aggressively
    reg_alpha=5.0,             # L1 Regularization (Lasso)
    reg_lambda=10.0,           # L2 Regularization (Ridge)
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=False
)
print("   * XGBoost Model trained successfully!")

# 5. Predictions & Evaluation
print("\n[Step 5] Evaluating Training vs Testing Accuracy...")

def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

train_preds = model.predict(X_train)
train_preds = np.clip(train_preds, 0, 500)
train_mae = mean_absolute_error(y_train, train_preds)
train_rmse = mean_squared_error(y_train, train_preds) ** 0.5
train_r2 = r2_score(y_train, train_preds)
train_mape = np.mean(np.abs((y_train - train_preds) / y_train)) * 100
train_smape = smape(y_train, train_preds)

test_preds = model.predict(X_test)
test_preds = np.clip(test_preds, 0, 500)  # AQI bounds
test_mae = mean_absolute_error(y_test, test_preds)
test_rmse = mean_squared_error(y_test, test_preds) ** 0.5
test_r2 = r2_score(y_test, test_preds)
test_mape = np.mean(np.abs((y_test - test_preds) / y_test)) * 100
test_smape = smape(y_test, test_preds)

print(f"\n   1. PRIMARY REGRESSION & TIME-SERIES METRICS")
print(f"   +----------+-------------------+-------------------+")
print(f"   | Metric   | Training Data     | Testing Data      |")
print(f"   +----------+-------------------+-------------------+")
print(f"   |  MAE     | {train_mae:>8.2f} AQI units | {test_mae:>8.2f} AQI units |")
print(f"   |  RMSE    | {train_rmse:>8.2f}           | {test_rmse:>8.2f}           |")
print(f"   |  R²      | {train_r2:>8.3f}           | {test_r2:>8.3f}           |")
print(f"   |  MAPE    | {train_mape:>7.2f}%          | {test_mape:>7.2f}%          |")
print(f"   |  SMAPE   | {train_smape:>7.2f}%          | {test_smape:>7.2f}%          |")
print(f"   +----------+-------------------+-------------------+")

# 5.5 Categorical Matrix Evaluation
print("\n   2. CATEGORICAL / CLASSIFICATION METRICS")
def get_aqi_category(aqi):
    if aqi <= 50: return "Good"
    elif aqi <= 100: return "Satisfactory"
    elif aqi <= 200: return "Moderate"
    elif aqi <= 300: return "Poor"
    elif aqi <= 400: return "Very Poor"
    else: return "Severe"

y_train_cats = [get_aqi_category(val) for val in y_train]
train_preds_cats = [get_aqi_category(val) for val in train_preds]
y_test_cats = [get_aqi_category(val) for val in y_test]
test_preds_cats = [get_aqi_category(val) for val in test_preds]

labels = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
cm_train = confusion_matrix(y_train_cats, train_preds_cats, labels=labels)
cm_test = confusion_matrix(y_test_cats, test_preds_cats, labels=labels)

print("\n   [TESTING DATA] CONFUSION MATRIX:")
print("   +------------------+------+------+------+------+-------+------+")
print("   |                  | Predicted Categories                     |")
print("   | Actual Categories| Good | Sat. | Mod. | Poor | V.Poor| Sev. |")
print("   +------------------+------+------+------+------+-------+------+")
for i, row_label in enumerate(["Good", "Sat.", "Mod.", "Poor", "V.Poor", "Sev."]):
    print(f"   | {row_label:<16} | {cm_test[i][0]:>4} | {cm_test[i][1]:>4} | {cm_test[i][2]:>4} | {cm_test[i][3]:>4} | {cm_test[i][4]:>5} | {cm_test[i][5]:>4} |")
print("   +------------------+------+------+------+------+-------+------+")

print("\n   [TESTING DATA] CLASSIFICATION REPORT (Precision, Recall, F1-Score):")
print(classification_report(y_test_cats, test_preds_cats, labels=[l for l in labels if l in y_test_cats], zero_division=0))


# 6. Feature Importance
print("\n[Step 6] Analyzing Feature Importance...")
importance = model.feature_importances_
imp_df = pd.DataFrame({'Feature': features, 'Importance': importance})
imp_df = imp_df.sort_values(by='Importance', ascending=False).head(10)
print("   Top 5 Most Important Features driving AQI:")
for idx, row in imp_df.head(5).iterrows():
    print(f"   - {row['Feature']:<20}: {row['Importance']:.4f}")

# 7. Visualization
print("\n[Step 7] Generating Visualizations...")

plt.style.use("dark_background")
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "figure.facecolor": "#0f1117",
    "axes.facecolor": "#1a1d27",
    "grid.color": "#2e3250",
})

# Actual vs Predicted Plot
fig, ax = plt.subplots(figsize=(15, 6))
ax.plot(test['Date'], y_test, color="#4fc3f7", linewidth=2, label="Actual Real AQI", marker="o", markersize=3)
ax.plot(test['Date'], test_preds, color="#ff7043", linewidth=2, linestyle="--", label="XGBoost Prediction", marker="s", markersize=3)

ax.fill_between(test['Date'], y_test, test_preds, alpha=0.15, color="#ffb300", label="Prediction Error")
ax.set_title(f"XGBoost: Actual vs Predicted AQI (Last 180 Days)\nTest MAE: {test_mae:.1f} | Test R²: {test_r2:.3f}", fontsize=14, pad=12)
ax.set_ylabel("Air Quality Index (AQI)")
ax.legend()
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("xgb_actual_vs_predicted.png", dpi=150, facecolor=fig.get_facecolor())
plt.close()
print("   * Saved: xgb_actual_vs_predicted.png")

# Feature Importance Plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(imp_df['Feature'][::-1], imp_df['Importance'][::-1], color="#ce93d8")
ax.set_title("XGBoost Feature Importance (Top 10)", fontsize=14, pad=12)
ax.set_xlabel("Relative Importance")
plt.tight_layout()
plt.savefig("xgb_feature_importance.png", dpi=150, facecolor=fig.get_facecolor())
plt.close()
print("   * Saved: xgb_feature_importance.png")

print("\n==================================================")
print("  * ML PIPELINE COMPLETE")
print("==================================================")
