import pandas as pd
import yaml
from sklearn.preprocessing import LabelEncoder
import os
import re

params = yaml.safe_load(open('params.yaml'))

os.makedirs('data/processed', exist_ok=True)

df = pd.read_csv('data/zameen-updated.csv')

print(f"Initial dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

def clean_area(area_val):
    """Extract numeric value from area string"""
    if pd.isna(area_val):
        return None
    
    area_str = str(area_val).strip()
    
    numbers = re.findall(r'\d+\.?\d*', area_str)
    
    if numbers:
        value = float(numbers[0])
        
        if value < 50 and 'marla' in area_str.lower():
            return value * 272
        elif value < 10 and 'kanal' in area_str.lower():
            return value * 5445
        
        return value
    return None

df['area_clean'] = df['area'].apply(clean_area)

def clean_numeric(val):
    """Clean any numeric column"""
    if pd.isna(val):
        return None
    
    if isinstance(val, (int, float)):
        return float(val)
    
    val_str = str(val).strip()
    
    val_str = val_str.replace(',', '')
    
    numbers = re.findall(r'\d+\.?\d*', val_str)
    
    if numbers:
        base_value = float(numbers[0])
        
        if 'crore' in val_str.lower():
            return base_value * 10000000
        elif 'lakh' in val_str.lower():
            return base_value * 100000
        
        return base_value
    return None

df['price_clean'] = df['price'].apply(clean_numeric)
df['bedrooms_clean'] = df['bedrooms'].apply(clean_numeric)
df['baths_clean'] = df['baths'].apply(clean_numeric)

print(f"\nCleaned numeric columns")

if 'purpose' in df.columns:
    df = df[df['purpose'] == 'For Sale'].copy()
    print(f"After filtering 'For Sale': {len(df)} records")

initial_count = len(df)
df = df.dropna(subset=['price_clean', 'area_clean', 'location'])
print(f"After removing NaN: {len(df)} records (removed {initial_count - len(df)})")

initial_count = len(df)
df = df[(df['price_clean'] > 500000) & (df['price_clean'] < 500000000)]
print(f"After price filter: {len(df)} records (removed {initial_count - len(df)})")

initial_count = len(df)
df = df[(df['area_clean'] > 50) & (df['area_clean'] < 20000)]
print(f"After area filter: {len(df)} records (removed {initial_count - len(df)})")

df['bedrooms_clean'] = df['bedrooms_clean'].fillna(df['bedrooms_clean'].median())
df['baths_clean'] = df['baths_clean'].fillna(df['baths_clean'].median())

df = df[(df['bedrooms_clean'] >= 0) & (df['bedrooms_clean'] <= 15)]

df = df[(df['baths_clean'] >= 0) & (df['baths_clean'] <= 10)]

print(f"After bedroom/bath filter: {len(df)} records")

if len(df) < 100:
    print(f"\nWARNING: Only {len(df)} records remaining!")
    print("This might not be enough for training. Consider:")
    print("1. Relaxing filters further")
    print("2. Checking data quality")
    print("3. Using a different dataset")

location_counts = df['location'].value_counts()
valid_locations = location_counts[location_counts >= 5].index
df = df[df['location'].isin(valid_locations)]

print(f"After filtering rare locations: {len(df)} records")

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

if len(df_final) > 0:
    print(f"\nFeature statistics:")
    print(df_final.describe())
    print(f"\nPrice range: PKR {df_final['price'].min():,.0f} - PKR {df_final['price'].max():,.0f}")
    print(f"Area range: {df_final['area'].min():.0f} - {df_final['area'].max():.0f} sq ft")
    print(f"Bedrooms range: {df_final['bedrooms'].min():.0f} - {df_final['bedrooms'].max():.0f}")
    print(f"Baths range: {df_final['baths'].min():.0f} - {df_final['baths'].max():.0f}")
    print(f"\nSample processed data:")
    print(df_final.head(20))
else:
    print("\nERROR: No data remaining after filtering!")
    print("Check your dataset and filtering logic.")