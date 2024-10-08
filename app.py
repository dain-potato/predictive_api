from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the trained model
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

    # Make a prediction using the model
    prediction = model.predict([[repair_count, average_cost, time_between_repairs]])

    # Return the prediction as a JSON response
    return jsonify({'prediction': prediction[0]})

# Start the Flask server
if __name__ == '__main__':
    app.run(debug=True)

@app.route('/')
def index():
    return "Flask app is running!"
