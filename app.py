from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('linear_regression_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    population = float(data['population'])
    rain = float(data['rain'])

    # Create a feature array for prediction
    features = np.array([[population, rain]])
    
    # Make prediction using the loaded model
    prediction = model.predict(features)

    # Return prediction result
    result = 'Drought' if prediction[0] else 'No Drought'
    return render_template('index.html', prediction_text=f'The model predicts: {result}')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json()
    population = data['population']
    rain = data['rain']

    # Create a feature array for prediction
    features = np.array([[population, rain]])

    # Make prediction using the loaded model
    prediction = model.predict(features)
    
    # Return prediction result as JSON
    result = {'prediction': 'Drought' if prediction[0] else 'No Drought'}
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)


