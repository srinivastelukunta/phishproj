
from flask import Flask, request, jsonify
import numpy as np
from model import load_model, predict

app = Flask(__name__)

# Load the pre-trained model
model, scaler = load_model()

@app.route('/')
def index():
    return "Welcome to the Phishing Detection API!"

@app.route('/predict', methods=['POST'])
def predict_phishing():
    try:
        # Get the input data from the POST request
        data = request.get_json()
        
        # Convert the input data into numpy array for prediction
        input_data = np.array(data['input']).reshape(1, -1)
        
        # Get prediction from the model
        prediction = predict(model, scaler, input_data)
        
        # Return the prediction result
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
