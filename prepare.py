import pandas as pd
import yaml
from sklearn.preprocessing import LabelEncoder
import os

params = yaml.safe_load(open('params.yaml'))

os.makedirs('data/processed', exist_ok=True)

df = pd.read_csv('data/zameen-updated.csv')

print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

df = df[df['purpose'] == 'For Sale'].copy()

df = df.dropna(subset=['price', 'area', 'bedrooms', 'baths', 'location'])

df = df[(df['price'] > 1000000) & (df['price'] < 100000000)]

df = df[(df['area'] > 100) & (df['area'] < 10000)]

le = LabelEncoder()
df['location_encoded'] = le.fit_transform(df['location'])

import joblib
os.makedirs('model', exist_ok=True)
joblib.dump(le, 'model/location_encoder.pkl')

features = ['area', 'bedrooms', 'baths', 'location_encoded']
df_processed = df[features + ['price']]

df_processed = df_processed.dropna()

df_processed.to_csv('data/processed/processed_data.csv', index=False)

print(f"\nProcessed {len(df_processed)} records")
print(f"Features: {features}")
print(f"Price range: {df_processed['price'].min():,.0f} - {df_processed['price'].max():,.0f}")
print(f"\nSample data:")
print(df_processed.head())