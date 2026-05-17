import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.ensemble import VotingRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             confusion_matrix, classification_report, f1_score)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

print("==================================================")
print("  TWO-STAGE ENSEMBLE AQI (ALL 5 FIXES APPLIED)")
print("==================================================")

# ── Data Loading ──────────────────────────────────────────────────────────────
print("\n[Step 1] Loading Dataset & Fixing Leakage...")
df = pd.read_csv("Final_Dataset.csv")
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# Fix Temporal Leakage
for col in ['AQI_roll_mean_7', 'AQI_ewma_7', 'AQI_diff_1']:
    df[col] = df[col].shift(1)

# ── Feature Engineering ───────────────────────────────────────────────────────
print("\n[Step 1.5] Engineering Advanced Features...")

# Lag features (all properly shifted)
for i in [2, 3, 7, 14]:
    df[f'AQI_lag_{i}'] = df['AQI'].shift(i)

# Rolling statistics
for w in [3, 7, 14]:
    df[f'AQI_roll_mean_{w}'] = df['AQI'].rolling(w).mean().shift(1)
    df[f'AQI_roll_std_{w}']  = df['AQI'].rolling(w).std().shift(1)
    df[f'AQI_roll_max_{w}']  = df['AQI'].rolling(w).max().shift(1)

# Meteorological Features
df['ventilation']      = df['windspeed'] / (df['humidity'] + 1)
df['ventilation_lag1'] = df['ventilation'].shift(1)
df['windspeed_lag1']   = df['windspeed'].shift(1)
df['humidity_lag1']    = df['humidity'].shift(1)
df['temp_lag1']        = df['temp'].shift(1)

# ── FIX 3: Pollution Spike Detector ──────────────────────────────────────────
# is_spike = 1 if AQI jumped >50 units from previous day (shifted to avoid leakage)
df['AQI_delta_1']  = df['AQI'].diff().shift(1)   # yesterday's daily change
df['is_spike']     = (df['AQI_delta_1'].abs() > 50).astype(int)
df['AQI_lag_2']    = df['AQI'].shift(2)
df['AQI_lag_3']    = df['AQI'].shift(3)

df = df.dropna().reset_index(drop=True)

# ── Target Encoding ───────────────────────────────────────────────────────────
def get_cat(aqi):
    if aqi <= 50:    return 0   # Good
    elif aqi <= 100: return 1   # Satisfactory
    elif aqi <= 200: return 2   # Moderate
    elif aqi <= 300: return 3   # Poor
    elif aqi <= 400: return 4   # Very Poor
    else:            return 5   # Severe

df['AQI_Category'] = df['AQI'].apply(get_cat)
features = [c for c in df.columns if c not in ['Date', 'index', 'AQI', 'AQI_Category']]

# ── Train / Val / Test Split ──────────────────────────────────────────────────
# Use chronological split: last 15% = test, previous 15% = val for calibration
n = len(df)
test_start = int(n * 0.85)
val_start  = int(n * 0.70)

train_df = df.iloc[:val_start].copy()
val_df   = df.iloc[val_start:test_start].copy()
test_df  = df.iloc[test_start:].copy()

X_train, y_train_reg, y_train_clf = train_df[features], train_df['AQI'], train_df['AQI_Category']
X_val,   y_val_reg,   y_val_clf   = val_df[features],   val_df['AQI'],   val_df['AQI_Category']
X_test,  y_test_reg,  y_test_clf  = test_df[features],  test_df['AQI'],  test_df['AQI_Category']

# ── FIX 4: Custom Asymmetric Loss for XGBoost ─────────────────────────────────
# Under-prediction is 2× more costly than over-prediction
def asymmetric_loss(y_true, y_pred):
    residual = y_true - y_pred
    grad = np.where(residual > 0, -2.0 * residual, -residual)
    hess = np.where(residual > 0, 2.0, 1.0)
    return grad, hess

# ── Stage 1: Ensemble Regressor ───────────────────────────────────────────────
print("\n[Step 2] Training Stage 1: Ensemble Regressor...")
y_train_log = np.log1p(y_train_reg)
y_val_log   = np.log1p(y_val_reg)
y_all_log   = np.log1p(pd.concat([y_train_reg, y_val_reg]))
X_all       = pd.concat([X_train, X_val])
y_all_clf   = pd.concat([y_train_clf, y_val_clf])

lgbm_reg = lgb.LGBMRegressor(
    n_estimators=1000, learning_rate=0.01, max_depth=7, num_leaves=31,
    subsample=0.8, colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=2.0,
    random_state=42, n_jobs=-1, verbose=-1
)
xgb_reg = xgb.XGBRegressor(
    n_estimators=1000, learning_rate=0.01, max_depth=6,
    subsample=0.8, colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=2.0,
    objective=asymmetric_loss,   # FIX 4 applied here
    random_state=42, n_jobs=-1, verbosity=0
)
ridge_reg = Ridge(alpha=10.0)

ensemble_reg = VotingRegressor([('lgbm', lgbm_reg), ('xgb', xgb_reg), ('ridge', ridge_reg)])
ensemble_reg.fit(X_all, y_all_log)   # train on train+val combined

train_preds_num = np.expm1(ensemble_reg.predict(X_train))
val_preds_num   = np.expm1(ensemble_reg.predict(X_val))
test_preds_num  = np.expm1(ensemble_reg.predict(X_test))

# Regression Metrics
def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

test_mae   = mean_absolute_error(y_test_reg, test_preds_num)
test_rmse  = mean_squared_error(y_test_reg,  test_preds_num) ** 0.5
test_r2    = r2_score(y_test_reg, test_preds_num)
test_mape  = np.mean(np.abs((y_test_reg - test_preds_num) / y_test_reg)) * 100
test_smape = smape(y_test_reg, test_preds_num)
train_r2   = r2_score(y_train_reg, train_preds_num)

print(f"\n   1. PRIMARY REGRESSION & TIME-SERIES METRICS")
print(f"   +----------+-------------------+-------------------+")
print(f"   | Metric   | Training Data     | Testing Data      |")
print(f"   +----------+-------------------+-------------------+")
print(f"   |  MAE     | {mean_absolute_error(y_train_reg,train_preds_num):>8.2f} AQI units | {test_mae:>8.2f} AQI units |")
print(f"   |  RMSE    | {mean_squared_error(y_train_reg,train_preds_num)**0.5:>8.2f}           | {test_rmse:>8.2f}           |")
print(f"   |  R\u00b2      | {train_r2:>8.3f}           | {test_r2:>8.3f}           |")
print(f"   |  MAPE    | {np.mean(np.abs((y_train_reg-train_preds_num)/y_train_reg))*100:>7.2f}%          | {test_mape:>7.2f}%          |")
print(f"   |  SMAPE   | {smape(y_train_reg,train_preds_num):>7.2f}%          | {test_smape:>7.2f}%          |")
print(f"   +----------+-------------------+-------------------+")

# ── Stage 2 Setup — Add Predicted AQI and Alarm Features ─────────────────────
X_train_clf = X_train.copy()
X_val_clf   = X_val.copy()
X_test_clf  = X_test.copy()

for src, preds in [(X_train_clf, train_preds_num), (X_val_clf, val_preds_num), (X_test_clf, test_preds_num)]:
    src['Predicted_AQI']       = preds
    src['alarm_predicted_poor'] = (preds >= 200).astype(int)
    src['alarm_lag_poor']       = (src['AQI_lag_1'] >= 200).astype(int)

# ── FIX 1: Asymmetric Sample Weights ─────────────────────────────────────────
print("\n[Step 3] Applying Fix 1: Asymmetric Cost Sample Weights...")
category_base_weight = {0: 1.0, 1: 1.5, 2: 2.0, 3: 4.0, 4: 5.0, 5: 8.0}

def compute_sample_weights(y_true, y_pred_cat):
    weights = []
    for actual, pred in zip(y_true, y_pred_cat):
        w = category_base_weight[actual]
        if pred < actual:  # Under-prediction: double penalty
            w *= 2.0
        weights.append(w)
    return np.array(weights)

# For initial training, use balanced class weights + base category scaling
classes = np.unique(y_train_clf)
bw = compute_class_weight('balanced', classes=classes, y=y_train_clf)
cw = {c: w * category_base_weight[c] for c, w in zip(classes, bw)}

print("\n[Step 4] Training Stage 2 Classifier...")

base_clf = lgb.LGBMClassifier(
    n_estimators=500, learning_rate=0.02, max_depth=6, num_leaves=31,
    min_child_samples=15, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=1.0, reg_lambda=2.0,
    class_weight=cw, random_state=42, n_jobs=-1, verbose=-1
)
base_clf.fit(X_train_clf, y_train_clf)

# ── FIX 5: Isotonic Probability Calibration ───────────────────────────────────
print("\n[Step 5] Applying Fix 5: Isotonic Probability Calibration...")
calibrated_clf = CalibratedClassifierCV(base_clf, method='isotonic', cv='prefit')
calibrated_clf.fit(X_val_clf, y_val_clf)

# ── FIX 2: Per-Boundary Threshold Tuning ─────────────────────────────────────
print("\n[Step 6] Applying Fix 2: Per-Boundary Threshold Tuning...")
val_probs = calibrated_clf.predict_proba(X_val_clf)

# Tune one threshold per adjacent class boundary
boundaries = [
    (1, 2, "Satisfactory -> Moderate"),
    (2, 3, "Moderate -> Poor"),
    (3, 4, "Poor -> Very Poor"),
    (4, 5, "Very Poor -> Severe"),
]

best_thresholds = {}
for low, high, name in boundaries:
    best_t, best_f1 = 0.5, 0
    for t in np.linspace(0.05, 0.95, 90):
        temp_preds = np.argmax(val_probs, axis=1)
        mask = (temp_preds == low) | (temp_preds == high)
        if mask.sum() == 0:
            continue
        rel = val_probs[mask, high] / (val_probs[mask, low] + val_probs[mask, high] + 1e-9)
        temp_preds[mask] = np.where(rel > t, high, low)
        # Maximize harmonic mean of F1 for both boundary classes
        f1_low  = f1_score((y_val_clf == low).astype(int),  (temp_preds == low).astype(int),  zero_division=0)
        f1_high = f1_score((y_val_clf == high).astype(int), (temp_preds == high).astype(int), zero_division=0)
        avg = (f1_low + f1_high) / 2
        if avg > best_f1:
            best_f1, best_t = avg, t
    best_thresholds[(low, high)] = best_t
    print(f"   * {name}: optimal threshold = {best_t:.3f}")


def apply_per_boundary_thresholds(probs, thresholds):
    preds = np.argmax(probs, axis=1)
    for (low, high), t in thresholds.items():
        mask = (preds == low) | (preds == high)
        if mask.sum() == 0:
            continue
        rel = probs[mask, high] / (probs[mask, low] + probs[mask, high] + 1e-9)
        preds[mask] = np.where(rel > t, high, low)
    return preds

# Final Test Predictions
test_probs    = calibrated_clf.predict_proba(X_test_clf)
test_preds_cat = apply_per_boundary_thresholds(test_probs, best_thresholds)

# ── Evaluation ────────────────────────────────────────────────────────────────
print("\n[Step 7] Evaluating Final Results...")
label_names = {0:"Good", 1:"Satisfactory", 2:"Moderate", 3:"Poor", 4:"Very Poor", 5:"Severe"}
y_test_str  = [label_names[v] for v in y_test_clf]
pred_str    = [label_names[v] for v in test_preds_cat]
labels_str  = ["Good","Satisfactory","Moderate","Poor","Very Poor","Severe"]

cm = confusion_matrix(y_test_str, pred_str, labels=labels_str)

print("\n   2. CATEGORICAL / CLASSIFICATION METRICS")
print("\n   [TESTING DATA] CONFUSION MATRIX:")
print("   +------------------+------+------+------+------+-------+------+")
print("   |                  | Predicted Categories                     |")
print("   | Actual Categories| Good | Sat. | Mod. | Poor | V.Poor| Sev. |")
print("   +------------------+------+------+------+------+-------+------+")
for i, row_label in enumerate(["Good","Sat.","Mod.","Poor","V.Poor","Sev."]):
    print(f"   | {row_label:<16} | {cm[i][0]:>4} | {cm[i][1]:>4} | {cm[i][2]:>4} | {cm[i][3]:>4} | {cm[i][4]:>5} | {cm[i][5]:>4} |")
print("   +------------------+------+------+------+------+-------+------+")

print("\n   [TESTING DATA] CLASSIFICATION REPORT:")
print(classification_report(y_test_str, pred_str,
                             labels=[l for l in labels_str if l in y_test_str],
                             zero_division=0))

# ── Visualization ─────────────────────────────────────────────────────────────
print("\n[Step 8] Generating Visualizations...")
plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(15, 6))
ax.plot(test_df['Date'], y_test_reg,     color="#4fc3f7", linewidth=2, label="Actual AQI",            marker="o", markersize=3)
ax.plot(test_df['Date'], test_preds_num, color="#ff7043", linewidth=2, label="Ensemble Prediction",    marker="s", markersize=3, linestyle="--")
ax.fill_between(test_df['Date'], y_test_reg, test_preds_num, alpha=0.15, color="#ffb300", label="Error Band")
ax.set_title(f"Ensemble (All 5 Fixes): Actual vs Predicted AQI\n"
             f"Test R\u00b2: {test_r2:.3f} | MAE: {test_mae:.1f} | MAPE: {test_mape:.2f}%",
             fontsize=13, pad=12)
ax.set_ylabel("Air Quality Index (AQI)")
ax.legend()
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("ensemble_actual_vs_predicted.png", dpi=150, facecolor=fig.get_facecolor())
plt.close()
print("   * Saved: ensemble_actual_vs_predicted.png")

print("\n==================================================")
print("  * PIPELINE COMPLETE (FIXES 1-5 APPLIED)")
print("==================================================")

# ── Save All Model Components to Disk ────────────────────────────────────────
import joblib
import os

os.makedirs("aqi_model_saved", exist_ok=True)

# Correct variable names from this script:
#   ensemble_reg    -> Stage 1 Ensemble Regressor
#   calibrated_clf  -> Stage 2 Isotonic-Calibrated Classifier
#   best_thresholds -> Per-boundary threshold dictionary (Fix 2)
#   features        -> List of feature column names

joblib.dump(ensemble_reg,     "aqi_model_saved/stage1_regressor.pkl")
joblib.dump(calibrated_clf,   "aqi_model_saved/stage2_calibrated.pkl")
joblib.dump(best_thresholds,  "aqi_model_saved/thresholds.pkl")
joblib.dump(features,         "aqi_model_saved/feature_columns.pkl")

print("=" * 50)
print("  ALL MODEL FILES SAVED TO: aqi_model_saved/")
print("  Files saved:")
print("  - stage1_regressor.pkl  (regression model)")
print("  - stage2_calibrated.pkl (classifier)")
print("  - thresholds.pkl        (boundary thresholds)")
print("  - feature_columns.pkl   (feature list)")
print("=" * 50)

# ── Loader Function (run this in any future script to skip retraining) ────────
def load_saved_model():
    stage1   = joblib.load("aqi_model_saved/stage1_regressor.pkl")
    clf      = joblib.load("aqi_model_saved/stage2_calibrated.pkl")
    thresh   = joblib.load("aqi_model_saved/thresholds.pkl")
    feats    = joblib.load("aqi_model_saved/feature_columns.pkl")
    print("Model loaded successfully!")
    return stage1, clf, thresh, feats

# Test save worked by reloading immediately
stage1_loaded, clf_loaded, thresh_loaded, features_loaded = load_saved_model()
print("Save + reload verified! Model is ready for deployment.")
