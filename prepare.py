import pandas as pd
import yaml
from sklearn.preprocessing import LabelEncoder
import os

params = yaml.safe_load(open('params.yaml'))

os.makedirs('data/processed', exist_ok=True)

df = pd.read_csv('data/zameen-updated.csv')

le = LabelEncoder()
df['location_encoded'] = le.fit_transform(df['location'])

features = ['property_size', 'bedrooms', 'bathrooms', 'location_encoded']
df_processed = df[features + ['price']]

df_processed = df_processed.dropna()

df_processed.to_csv('data/processed/processed_data.csv', index=False)

print(f"Processed {len(df_processed)} records")
print(f"Features: {features}")