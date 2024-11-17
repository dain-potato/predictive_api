from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
CORS(app)

# Load the trained model
# model = joblib.load('maintenance_model.pkl')
model_path = os.path.join(os.path.dirname(__file__), 'maintenance_model.pkl')
model = joblib.load(model_path)

# Define a route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get the request data (JSON)
    data = request.get_json()

    # Extract features from the request
    features = [
        data['purchase_cost'],
        data['lifespan_years'],
        data['salvage_value'],
        data['depreciation_per_year'],
        data['repair_count'],
        data['average_cost'],
        data['time_between_repairs']
    ]

    # Make a prediction using the model
    prediction = model.predict([features])

    # Return the prediction as a JSON response
    return jsonify({'prediction': prediction[0]})

# Define the root route
@app.route('/', methods=['GET'])
def index():
    return "Flask app is running!"

# Start the Flask server
if __name__ == '__main__':
    app.run(debug=True)
