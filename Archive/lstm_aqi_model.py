import pandas as pd
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF logging
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

print("==================================================")
print("  DEEP LEARNING: LSTM AQI PREDICTOR")
print("==================================================")

# 1. Load Data
print("\n[Step 1] Loading Dataset & Fixing Leakage...")
df = pd.read_csv("Final_Dataset.csv")
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

leaking_features = ['AQI_roll_mean_7', 'AQI_ewma_7', 'AQI_diff_1']
for col in leaking_features:
    df[col] = df[col].shift(1)

df = df.dropna().reset_index(drop=True)

# 2. Features and Target
target = 'AQI'
features = [col for col in df.columns if col not in ['Date', 'index', 'AQI']]

def get_aqi_category_num(aqi):
    if aqi <= 50: return 0
    elif aqi <= 100: return 1
    elif aqi <= 200: return 2
    elif aqi <= 300: return 3
    elif aqi <= 400: return 4
    else: return 5

df['AQI_Category'] = df['AQI'].apply(get_aqi_category_num)

# Split by Date (180 days for testing)
split_date = df['Date'].max() - pd.Timedelta(days=180)
train_df = df[df['Date'] <= split_date].copy()
test_df  = df[df['Date'] > split_date].copy()

# Ensure chronological order for LSTM
train_df = train_df.sort_values('Date')
test_df = test_df.sort_values('Date')

# 3. Scaling (Neural Networks REQUIRE scaling)
print("\n[Step 2] Scaling Features & Target for Neural Network...")
feature_scaler = MinMaxScaler(feature_range=(0, 1))
target_scaler = MinMaxScaler(feature_range=(0, 1))

train_features_scaled = feature_scaler.fit_transform(train_df[features])
test_features_scaled = feature_scaler.transform(test_df[features])

train_target_scaled = target_scaler.fit_transform(train_df[[target]])
test_target_scaled = target_scaler.transform(test_df[[target]])

# 4. Reshape Data for LSTM (Sliding Window)
# We will use the past TIME_STEPS days to predict the next day
TIME_STEPS = 7
print(f"\n[Step 3] Reshaping Data into 3D Tensors (Time Steps: {TIME_STEPS})...")

def create_sequences(X, y, time_steps):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

X_train_seq, y_train_seq = create_sequences(train_features_scaled, train_target_scaled, TIME_STEPS)
X_test_seq, y_test_seq = create_sequences(test_features_scaled, test_target_scaled, TIME_STEPS)

# Get the corresponding dates and actual AQI for the test set
test_dates = test_df['Date'].values[TIME_STEPS:]
test_actual_aqi = test_df['AQI'].values[TIME_STEPS:]
test_actual_cat = test_df['AQI_Category'].values[TIME_STEPS:]

print(f"   * Training Tensor Shape: {X_train_seq.shape}")
print(f"   * Testing Tensor Shape: {X_test_seq.shape}")

# 5. Build LSTM Architecture
print("\n[Step 4] Building LSTM Neural Network...")
model = Sequential()
model.add(LSTM(64, activation='tanh', return_sequences=False, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(1)) # Output is 1 numerical value

model.compile(optimizer='adam', loss='mse')

# 6. Train LSTM
print("\n[Step 5] Training LSTM Model (50 Epochs)...")
history = model.fit(
    X_train_seq, y_train_seq,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    verbose=0 # Suppress epoch spam
)
print("   * Training Complete!")

# 7. Predict & Inverse Transform
print("\n[Step 6] Predicting and Inverse Transforming...")
test_preds_scaled = model.predict(X_test_seq, verbose=0)
test_preds_num = target_scaler.inverse_transform(test_preds_scaled).flatten()

# Regression Metrics
def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

test_mae = mean_absolute_error(test_actual_aqi, test_preds_num)
test_rmse = mean_squared_error(test_actual_aqi, test_preds_num) ** 0.5
test_r2 = r2_score(test_actual_aqi, test_preds_num)
test_mape = np.mean(np.abs((test_actual_aqi - test_preds_num) / test_actual_aqi)) * 100
test_smape = smape(test_actual_aqi, test_preds_num)

print(f"\n   1. NUMERICAL PREDICTION (REGRESSION) METRICS")
print(f"   +----------+-------------------+")
print(f"   | Metric   | Testing Data      |")
print(f"   +----------+-------------------+")
print(f"   |  MAE     | {test_mae:>8.2f} AQI units |")
print(f"   |  RMSE    | {test_rmse:>8.2f}           |")
print(f"   |  R²      | {test_r2:>8.3f}           |")
print(f"   |  MAPE    | {test_mape:>7.2f}%          |")
print(f"   |  SMAPE   | {test_smape:>7.2f}%          |")
print(f"   +----------+-------------------+")

# 8. Categorical Mapping (Applying the thresholds)
print("\n[Step 7] Mapping numerical predictions to Health Categories...")
test_preds_cat_raw = [get_aqi_category_num(val) for val in test_preds_num]

label_names = {0: "Good", 1: "Satisfactory", 2: "Moderate", 3: "Poor", 4: "Very Poor", 5: "Severe"}
y_test_cats_str = [label_names[val] for val in test_actual_cat]
preds_cats_str = [label_names[val] for val in test_preds_cat_raw]
labels_str = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]

cm_test = confusion_matrix(y_test_cats_str, preds_cats_str, labels=labels_str)

print("\n   2. CLASSIFICATION METRICS")
print("\n   [TESTING DATA] CONFUSION MATRIX:")
print("   +------------------+------+------+------+------+-------+------+")
print("   |                  | Predicted Categories                     |")
print("   | Actual Categories| Good | Sat. | Mod. | Poor | V.Poor| Sev. |")
print("   +------------------+------+------+------+------+-------+------+")
for i, row_label in enumerate(["Good", "Sat.", "Mod.", "Poor", "V.Poor", "Sev."]):
    print(f"   | {row_label:<16} | {cm_test[i][0]:>4} | {cm_test[i][1]:>4} | {cm_test[i][2]:>4} | {cm_test[i][3]:>4} | {cm_test[i][4]:>5} | {cm_test[i][5]:>4} |")
print("   +------------------+------+------+------+------+-------+------+")

print("\n   [TESTING DATA] CLASSIFICATION REPORT:")
print(classification_report(y_test_cats_str, preds_cats_str, labels=[l for l in labels_str if l in y_test_cats_str], zero_division=0))

# 9. Visualization
print("\n[Step 8] Generating Visualizations...")
plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(15, 6))
ax.plot(test_dates, test_actual_aqi, color="#4fc3f7", linewidth=2, label="Actual Real AQI", marker="o", markersize=3)
ax.plot(test_dates, test_preds_num, color="#ff4081", linewidth=2, linestyle="--", label="LSTM Prediction", marker="s", markersize=3)
ax.fill_between(test_dates, test_actual_aqi, test_preds_num, alpha=0.15, color="#ffb300", label="Prediction Error")
ax.set_title(f"Deep Learning (LSTM): Actual vs Predicted AQI\nTest MAE: {test_mae:.1f} | Test MAPE: {test_mape:.2f}%", fontsize=14, pad=12)
ax.set_ylabel("Air Quality Index (AQI)")
ax.legend()
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("lstm_actual_vs_predicted.png", dpi=150, facecolor=fig.get_facecolor())
plt.close()
print("   * Saved: lstm_actual_vs_predicted.png")

print("\n==================================================")
print("  * LSTM DEEP LEARNING MODEL COMPLETE")
print("==================================================")
