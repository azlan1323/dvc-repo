import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import json
import os
import numpy as np

params = yaml.safe_load(open('params.yaml'))

os.makedirs('model', exist_ok=True)
os.makedirs('metrics', exist_ok=True)

df = pd.read_csv('data/processed/processed_data.csv')

print(f"Training data shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

if len(df) < 10:
    print(f"\nERROR: Only {len(df)} samples available!")
    print("Need at least 10 samples for training. Check your data preprocessing.")
    exit(1)

X = df.drop('price', axis=1)
y = df['price']

print(f"\nFeature columns: {X.columns.tolist()}")

test_size = params['data']['test_size']
if len(df) < 50:
    test_size = 0.3
    print(f"Small dataset detected. Using test_size={test_size}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=test_size,
    random_state=params['data']['random_state']
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

n_estimators = params['model']['n_estimators']
if len(df) < 100:
    n_estimators = min(50, n_estimators)
    print(f"Using {n_estimators} estimators for small dataset")

model = RandomForestRegressor(
    n_estimators=n_estimators,
    max_depth=params['model']['max_depth'],
    min_samples_split=max(2, min(params['model']['min_samples_split'], len(X_train) // 10)),
    random_state=params['model']['random_state'],
    n_jobs=-1
)

print("\nTraining model...")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

mape = np.mean(np.abs((y_test - y_pred) / np.maximum(y_test, 1))) * 100

joblib.dump(model, 'model/model.pkl')

metrics = {
    'mse': float(mse),
    'rmse': float(rmse),
    'mae': float(mae),
    'r2_score': float(r2),
    'mape': float(mape),
    'n_samples': len(df),
    'n_train': len(X_train),
    'n_test': len(X_test)
}

with open('metrics/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4)

print(f"\nModel trained successfully!")
print(f"Metrics:")
print(f"   RMSE: PKR {rmse:,.0f}")
print(f"   MAE: PKR {mae:,.0f}")
print(f"   R² Score: {r2:.4f}")
print(f"   MAPE: {mape:.2f}%")