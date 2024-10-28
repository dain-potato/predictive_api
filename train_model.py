import pandas as pd
import mysql.connector
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# TRAIN THE MODEL EVERY 3 MONTHS OR 6 MONTHS OR YEAR (DEPENDS)
# Step 1: Connect to the MySQL database
db_connection = mysql.connector.connect(
    host='165.22.59.23',  # XAMPP default host
    user='root',  # XAMPP default user
    password='password',  # Leave password blank unless you have set one
    database='fixedasset'  # Your database name
)

# Step 2: Query the maintenance data
query = """
    SELECT asset_key, 
           COUNT(id) as repair_count, 
           AVG(cost) as average_cost, 
           DATEDIFF(MAX(completion_date), MIN(start_date)) as time_between_repairs
    FROM maintenance
    WHERE completed = 1  -- Only use completed maintenance records
    GROUP BY asset_key
"""
df = pd.read_sql(query, db_connection)

# Step 3: Close the database connection
db_connection.close()

# Step 4: Define the features (X)
X = df[['repair_count', 'average_cost', 'time_between_repairs']]  # Features

# Step 5: Create manual labels for training (temporary solution)
# You can modify this logic based on domain knowledge or after more data analysis.
def label_data(row):
    if row['repair_count'] > 5:
        return 'dispose'
    elif row['average_cost'] > 5000:
        return 'repair'
    else:
        return 'maintenance'

# Step 6: Apply the manual labeling function to create labels
df['recommendation'] = df.apply(label_data, axis=1)

# Step 7: Define the target variable (y)
y = df['recommendation']  # Target variable (generated manually)

# Step 8: Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 9: Initialize and train the decision tree model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Step 10: Evaluate the model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Model accuracy: {accuracy}")

# Step 11: Save the trained model to a file
joblib.dump(model, 'maintenance_model.pkl')
