from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('phishing_detector.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        data = request.get_json()

        # Extract features from the JSON data
        features = data['features']

        # Convert features to a numpy array and reshape for prediction
        input_data = np.array(features).reshape(1, -1)

        # Make prediction
        prediction = model.predict(input_data)[0]

        # Encode prediction back to original labels if necessary
        # Assuming 1: Phishing, 0: Legitimate
        label = 'Phishing' if prediction == 1 else 'Legitimate'

        return jsonify({'prediction': label})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)