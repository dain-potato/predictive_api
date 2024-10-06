import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Step 1: Create some sample data (you can replace this with real maintenance data)
data = {
    'repair_count': [1, 2, 5, 6, 10],
    'average_cost': [1000, 1500, 4000, 6000, 8000],
    'time_between_repairs': [30, 45, 90, 100, 150],
    'recommendation': ['maintenance', 'maintenance', 'repair', 'repair', 'dispose']
}

# Step 2: Convert the data into a pandas DataFrame
df = pd.DataFrame(data)

# Step 3: Define the features (X) and the target variable (y)
X = df[['repair_count', 'average_cost', 'time_between_repairs']]  # Features
y = df['recommendation']  # Target variable

# Step 4: Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Initialize and train the decision tree model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Step 6: Evaluate the model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Model accuracy: {accuracy}")

# Step 7: Save the trained model to a file
joblib.dump(model, 'maintenance_model.pkl')
