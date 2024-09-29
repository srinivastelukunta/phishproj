# Comment for git push
from flask import Flask, request, jsonify
import joblib
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

# Load the trained model
model = joblib.load('phishing_detector.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        logging.info('Received a request')
        
        data = request.get_json()
        logging.info(f"Request JSON: {data}")

        if 'features' not in data:
            logging.error("Missing 'features' in request body")
            return jsonify({'error': "Missing 'features' in request body"}), 400

        # Extract features from the JSON data
        features = data['features']

        # Convert features to a numpy array and reshape for prediction
        input_data = np.array(features).reshape(1, -1)

        # Make prediction
        prediction = model.predict(input_data)[0]

        # Encode prediction back to original labels
        label = 'Phishing' if prediction == 1 else 'Legitimate'

        logging.info(f"Prediction: {label}")
        return jsonify({'prediction': label})

    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
