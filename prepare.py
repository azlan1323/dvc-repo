import pandas as pd
import yaml
from sklearn.preprocessing import LabelEncoder
import os
import re

params = yaml.safe_load(open('params.yaml'))

os.makedirs('data/processed', exist_ok=True)

df = pd.read_csv('data/zameen-updated.csv')

print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

print(f"\nData types:")
print(df[['price', 'area', 'bedrooms', 'baths']].dtypes)

print(f"\nSample area values:")
print(df['area'].head(10))

def clean_area(area_val):
    """Extract numeric value from area string"""
    if pd.isna(area_val):
        return None
    
    area_str = str(area_val)
    
    numbers = re.findall(r'\d+\.?\d*', area_str)
    
    if numbers:
        return float(numbers[0])
    return None

df['area_clean'] = df['area'].apply(clean_area)

def clean_numeric(val):
    """Clean any numeric column"""
    if pd.isna(val):
        return None
    try:
        return float(val)
    except:
        numbers = re.findall(r'\d+\.?\d*', str(val))
        if numbers:
            return float(numbers[0])
        return None

df['price_clean'] = df['price'].apply(clean_numeric)
df['bedrooms_clean'] = df['bedrooms'].apply(clean_numeric)
df['baths_clean'] = df['baths'].apply(clean_numeric)

print(f"\n✅ Cleaned numeric columns")
print(f"Sample cleaned area: {df['area_clean'].head(10).tolist()}")

if 'purpose' in df.columns:
    df = df[df['purpose'] == 'For Sale'].copy()
    print(f"After filtering 'For Sale': {len(df)} records")

df = df.dropna(subset=['price_clean', 'area_clean', 'bedrooms_clean', 'baths_clean', 'location'])
print(f"After removing NaN: {len(df)} records")

df = df[(df['price_clean'] > 1000000) & (df['price_clean'] < 100000000)]

df = df[(df['area_clean'] > 100) & (df['area_clean'] < 10000)]

df = df[(df['bedrooms_clean'] >= 1) & (df['bedrooms_clean'] <= 10)]

df = df[(df['baths_clean'] >= 1) & (df['baths_clean'] <= 8)]

print(f"After outlier removal: {len(df)} records")

le = LabelEncoder()
df['location_encoded'] = le.fit_transform(df['location'])

os.makedirs('model', exist_ok=True)
import joblib
joblib.dump(le, 'model/location_encoder.pkl')

print(f"Encoded {len(le.classes_)} unique locations")

df_final = pd.DataFrame({
    'area': df['area_clean'],
    'bedrooms': df['bedrooms_clean'],
    'baths': df['baths_clean'],
    'location_encoded': df['location_encoded'],
    'price': df['price_clean']
})

df_final = df_final.dropna()

df_final.to_csv('data/processed/processed_data.csv', index=False)

print(f"\nSuccessfully processed {len(df_final)} records")
print(f"\nFeature statistics:")
print(df_final.describe())
print(f"\nPrice range: PKR {df_final['price'].min():,.0f} - PKR {df_final['price'].max():,.0f}")
print(f"Area range: {df_final['area'].min():.0f} - {df_final['area'].max():.0f} sq ft")
print(f"\nSample processed data:")
print(df_final.head(10))