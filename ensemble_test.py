import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import r2_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

df = pd.read_csv('Final_Dataset.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# Fix leakage
for col in ['AQI_roll_mean_7', 'AQI_ewma_7', 'AQI_diff_1']:
    df[col] = df[col].shift(1)

# Advanced Features
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
df = df.dropna().reset_index(drop=True)

def get_cat(aqi):
    if aqi <= 50:   return 0
    elif aqi <= 100: return 1
    elif aqi <= 200: return 2
    elif aqi <= 300: return 3
    elif aqi <= 400: return 4
    else:            return 5

df['AQI_Category'] = df['AQI'].apply(get_cat)
features = [c for c in df.columns if c not in ['Date','index','AQI','AQI_Category']]

train, test = train_test_split(df, test_size=0.15, stratify=df['AQI_Category'], random_state=42)
test = test.sort_values('Date').reset_index(drop=True)

X_train, y_train     = train[features], train['AQI']
X_test,  y_test      = test[features],  test['AQI']
y_train_clf, y_test_clf = train['AQI_Category'], test['AQI_Category']

y_log = np.log1p(y_train)

# Stage 1: Ensemble Regressor (LightGBM + XGBoost + Ridge)
m1 = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.01, max_depth=7, num_leaves=31, subsample=0.8, colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=2.0, random_state=42, n_jobs=-1, verbose=-1)
m2 = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.01, max_depth=6, subsample=0.8, colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=2.0, random_state=42, n_jobs=-1, verbosity=0)
m3 = Ridge(alpha=10.0)

ensemble = VotingRegressor([('lgbm', m1), ('xgb', m2), ('ridge', m3)])
ensemble.fit(X_train, y_log)

tr_preds = np.expm1(ensemble.predict(X_train))
te_preds = np.expm1(ensemble.predict(X_test))

train_r2  = r2_score(y_train, tr_preds)
test_r2   = r2_score(y_test,  te_preds)
test_mape = np.mean(np.abs((y_test - te_preds) / y_test)) * 100
test_mae  = np.mean(np.abs(y_test - te_preds))
test_rmse = np.sqrt(np.mean((y_test - te_preds)**2))

print("\n=== ENSEMBLE REGRESSOR ===")
print(f"Train R2 : {train_r2:.4f}")
print(f"Test  R2 : {test_r2:.4f}")
print(f"MAE      : {test_mae:.2f}")
print(f"RMSE     : {test_rmse:.2f}")
print(f"MAPE     : {test_mape:.2f}%")

# Stage 2: Classifier with Tuned Weights
X_tr_clf = X_train.copy()
X_te_clf = X_test.copy()
X_tr_clf['Predicted_AQI'] = tr_preds
X_te_clf['Predicted_AQI'] = te_preds
X_tr_clf['alarm_poor'] = (tr_preds >= 200).astype(int)
X_te_clf['alarm_poor'] = (te_preds >= 200).astype(int)
X_tr_clf['alarm_lag']  = (X_train['AQI_lag_1'] >= 200).astype(int)
X_te_clf['alarm_lag']  = (X_test['AQI_lag_1'] >= 200).astype(int)

classes = np.unique(y_train_clf)
bw = compute_class_weight('balanced', classes=classes, y=y_train_clf)
cw = {c: w for c, w in zip(classes, bw)}
cw[2] *= 1.5
cw[3] *= 1.5
cw[5] *= 1.2

clf = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.02, max_depth=6, num_leaves=31, class_weight=cw, random_state=42, n_jobs=-1, verbose=-1)
clf.fit(X_tr_clf, y_train_clf)

probs    = clf.predict_proba(X_te_clf)
preds_cat = np.argmax(probs, axis=1)
mask     = (preds_cat == 2) | (preds_cat == 3)
rel      = probs[mask, 3] / (probs[mask, 2] + probs[mask, 3] + 1e-9)
preds_cat[mask] = np.where(rel > 0.585, 3, 2)

acc        = accuracy_score(y_test_clf, preds_cat)
f1_macro   = f1_score(y_test_clf, preds_cat, average='macro',    zero_division=0)
f1_weighted= f1_score(y_test_clf, preds_cat, average='weighted', zero_division=0)

print("\n=== ENSEMBLE CLASSIFIER ===")
print(f"Accuracy    : {acc:.4f}")
print(f"F1 Macro    : {f1_macro:.4f}")
print(f"F1 Weighted : {f1_weighted:.4f}")
