from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the saved machine learning model
model = joblib.load('maintenance_model.pkl')

# Define a route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get the request data (JSON)
    data = request.get_json()

    # Extract features from the request
    repair_count = data['repair_count']
    average_cost = data['average_cost']
    time_between_repairs = data['time_between_repairs']

    # Prepare the input data for the model (ensure it's in the correct format)
    input_data = np.array([[repair_count, average_cost, time_between_repairs]])

    # Use the model to make a prediction
    prediction = model.predict(input_data)[0]  # Get the first prediction

    # Return the prediction as a JSON response
    return jsonify({'prediction': prediction})

# Start the Flask server
if __name__ == '__main__':
    app.run(debug=True)
