from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

model_path = 'model/model.pkl'
encoder_path = 'model/location_encoder.pkl'

if os.path.exists(model_path) and os.path.exists(encoder_path):
    model = joblib.load(model_path)
    location_encoder = joblib.load(encoder_path)
    locations = location_encoder.classes_.tolist()
else:
    model = None
    location_encoder = None
    locations = []
    print("Model or encoder not found! Run 'dvc pull' to fetch them.")

@app.route('/')
def home():
    return render_template('index.html', locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or location_encoder is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        area = float(request.form['area'])
        bedrooms = int(request.form['bedrooms'])
        baths = int(request.form['baths'])
        location = request.form['location']
        
        location_encoded = location_encoder.transform([location])[0]
        
        features = np.array([[area, bedrooms, baths, location_encoded]])
        
        prediction = model.predict(features)[0]
        
        return render_template('index.html', 
                             locations=locations,
                             prediction_text=f'Predicted House Price: PKR {prediction:,.0f}',
                             input_area=area,
                             input_bedrooms=bedrooms,
                             input_baths=baths,
                             input_location=location)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/predict', methods=['POST'])
def api_predict():
    if model is None or location_encoder is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.json
        
        area = float(data['area'])
        bedrooms = int(data['bedrooms'])
        baths = int(data['baths'])
        location = data['location']
        
        location_encoded = location_encoder.transform([location])[0]
        features = np.array([[area, bedrooms, baths, location_encoded]])
        prediction = model.predict(features)[0]
        
        return jsonify({'predicted_price': float(prediction)})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)