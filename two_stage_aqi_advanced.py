import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, confusion_matrix, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

print("==================================================")
print("  TWO-STAGE AQI PREDICTION (ADVANCED ML + FINE-TUNING)")
print("==================================================")

# 1. Load Data
df = pd.read_csv("Final_Dataset.csv")
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# 1.5 FIX DATA LEAKAGE
leaking_features = ['AQI_roll_mean_7', 'AQI_ewma_7', 'AQI_diff_1']
for col in leaking_features:
    df[col] = df[col].shift(1)

# 1.6 ADD ADVANCED FEATURES
df['AQI_lag_14'] = df['AQI'].shift(14)
df['AQI_roll_std_7'] = df['AQI'].rolling(7).std().shift(1)
df['AQI_roll_max_7'] = df['AQI'].rolling(7).max().shift(1)
df = df.dropna().reset_index(drop=True)

# 2. Prepare Features and Target
features = [col for col in df.columns if col not in ['Date', 'index', 'AQI']]

def get_aqi_category_num(aqi):
    if aqi <= 50: return 0      # Good
    elif aqi <= 100: return 1   # Satisfactory
    elif aqi <= 200: return 2   # Moderate
    elif aqi <= 300: return 3   # Poor
    elif aqi <= 400: return 4   # Very Poor
    else: return 5              # Severe

df['AQI_Category'] = df['AQI'].apply(get_aqi_category_num)

# 3. Train / Test Split
train, test = train_test_split(df, test_size=0.15, stratify=df['AQI_Category'], random_state=42)
test = test.sort_values('Date').reset_index(drop=True)

X_train, y_train_reg, y_train_clf = train[features], train['AQI'], train['AQI_Category']
X_test, y_test_reg, y_test_clf   = test[features], test['AQI'], test['AQI_Category']

# ---------------------------------------------------------
# STAGE 1: REGRESSION
# ---------------------------------------------------------
print("\n[Step 1] Training Stage 1: Numerical Regressor...")
y_train_log = np.log1p(y_train_reg)

reg_model = lgb.LGBMRegressor(
    n_estimators=600,
    learning_rate=0.015,
    max_depth=5,
    num_leaves=18,
    min_child_samples=25,
    subsample=0.7,
    colsample_bytree=0.7,
    reg_alpha=1.0,
    reg_lambda=3.0,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)
reg_model.fit(X_train, y_train_log)

train_preds_num = np.expm1(reg_model.predict(X_train))
test_preds_num = np.expm1(reg_model.predict(X_test))

# ---------------------------------------------------------
# STAGE 2: CLASSIFICATION WITH FINE-TUNING
# ---------------------------------------------------------
print("\n[Step 2] Applying Fine-Tuned Feature Engineering...")

X_train_clf = X_train.copy()
X_test_clf = X_test.copy()

# Add Stage 1 Prediction as a feature
X_train_clf['Predicted_AQI'] = train_preds_num
X_test_clf['Predicted_AQI'] = test_preds_num

# 1. Targeted Feature Engineering: Boundary Alarms
# Note: Raw PM2.5/PM10 isn't available, so we engineer strict boundary alarms off the prediction and recent lags.
X_train_clf['alarm_predicted_poor'] = (X_train_clf['Predicted_AQI'] >= 200).astype(int)
X_test_clf['alarm_predicted_poor'] = (X_test_clf['Predicted_AQI'] >= 200).astype(int)

X_train_clf['alarm_lag_poor'] = (X_train_clf['AQI_lag_1'] >= 200).astype(int)
X_test_clf['alarm_lag_poor'] = (X_test_clf['AQI_lag_1'] >= 200).astype(int)

# 2. Custom Class Weights
print("\n[Step 3] Computing Custom Penalties for Moderate/Poor Errors...")
classes = np.unique(y_train_clf)
balanced_weights = compute_class_weight('balanced', classes=classes, y=y_train_clf)
custom_weights = {c: w for c, w in zip(classes, balanced_weights)}

# Amplify the penalty for the problematic classes
custom_weights[2] *= 1.5  # Moderate
custom_weights[3] *= 1.5  # Poor
custom_weights[5] *= 1.2  # Severe (to keep its recall high)

clf_model = lgb.LGBMClassifier(
    n_estimators=300,
    learning_rate=0.03,
    max_depth=5,
    num_leaves=20,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=1.0,
    reg_lambda=2.0,
    random_state=42,
    class_weight=custom_weights,
    n_jobs=-1,
    verbose=-1
)
clf_model.fit(X_train_clf, y_train_clf)

# 3. Custom Probability Thresholds
print("\n[Step 4] Optimizing Probability Boundary Post-Processing...")
train_probs = clf_model.predict_proba(X_train_clf)
best_t = 0.5
best_f1 = 0

# Test different probability thresholds specifically for deciding between Moderate (2) and Poor (3)
for t in np.linspace(0.1, 0.9, 90):
    temp_preds = np.argmax(train_probs, axis=1)
    mask = (temp_preds == 2) | (temp_preds == 3)
    relative_prob = train_probs[mask, 3] / (train_probs[mask, 2] + train_probs[mask, 3] + 1e-9)
    temp_preds[mask] = np.where(relative_prob > t, 3, 2)
    
    avg_f1 = (f1_score(y_train_clf == 2, temp_preds == 2) + f1_score(y_train_clf == 3, temp_preds == 3)) / 2
    if avg_f1 > best_f1:
        best_f1 = avg_f1
        best_t = t

print(f"   * Optimal Probability Shift (Moderate vs Poor): {best_t:.3f}")

# Apply the custom probability shift to the Test Set
test_probs = clf_model.predict_proba(X_test_clf)
test_preds_cat = np.argmax(test_probs, axis=1)
mask = (test_preds_cat == 2) | (test_preds_cat == 3)
relative_prob = test_probs[mask, 3] / (test_probs[mask, 2] + test_probs[mask, 3] + 1e-9)
test_preds_cat[mask] = np.where(relative_prob > best_t, 3, 2)

# Evaluation
print("\n[Step 5] Evaluating Final Classification...")
label_names = {0: "Good", 1: "Satisfactory", 2: "Moderate", 3: "Poor", 4: "Very Poor", 5: "Severe"}
y_test_cats_str = [label_names[val] for val in y_test_clf]
preds_cats_str = [label_names[val] for val in test_preds_cat]
labels_str = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]

cm_test = confusion_matrix(y_test_cats_str, preds_cats_str, labels=labels_str)

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

print("\n==================================================")
print("  * PIPELINE REFINEMENT COMPLETE")
print("==================================================")
