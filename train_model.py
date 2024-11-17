import pandas as pd
import mysql.connector
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from sklearn.utils import resample

# Connect to the MySQL database
try:
    db_connection = mysql.connector.connect(
      host='165.22.59.23',  # XAMPP default host
      user='laraveluser',  # XAMPP default user
      password='password',  # Leave password blank unless you have set one
      database='fixedasset'  # Your database name
    )
except mysql.connector.Error as err:
    print(f"Error: {err}")
    exit(1)

# Updated SQL query to include additional variables from the `asset` table
query = """
    SELECT 
        asset.id AS asset_key,
        asset.purchase_cost,
        asset.purchase_date,
        asset.usage_lifespan AS lifespan_years,
        asset.salvage_value,
        asset.depreciation AS depreciation_per_year,
        COUNT(maintenance.id) AS repair_count,
        AVG(maintenance.cost) AS average_cost,
        DATEDIFF(MAX(maintenance.completion_date), MIN(maintenance.start_date)) AS time_between_repairs
    FROM maintenance
    JOIN asset ON maintenance.asset_key = asset.id
    WHERE maintenance.is_completed = 1
    GROUP BY asset.id
"""

# Fetch data
df = pd.read_sql(query, db_connection)
db_connection.close()

# Handle missing values by filling with 0 (or you could choose to drop them if preferred)
df = df.fillna(0)

# Define features (X) with new columns
X = df[['purchase_cost', 'lifespan_years', 'salvage_value', 'depreciation_per_year',
        'repair_count', 'average_cost', 'time_between_repairs']]

# Updated labeling function with refined and more balanced criteria
def label_data(row):
    # If the asset is beyond its lifespan or has excessive repairs, suggest disposal
    if row['lifespan_years'] <= 0 or row['repair_count'] > 8 or row['salvage_value'] < (0.05 * row['purchase_cost']):
        return 'dispose'
    # If the average cost of maintenance is significantly high or repair count is moderate to high, suggest repair
    elif row['average_cost'] > (0.5 * row['purchase_cost']) or (5 <= row['repair_count'] <= 8):
        return 'repair'
    # If the repair count is low, average cost is manageable, and the asset is within its lifespan, suggest maintenance
    elif row['repair_count'] <= 4 and row['average_cost'] <= (0.3 * row['purchase_cost']) and row['lifespan_years'] > 0:
        return 'maintenance'
    # Default to maintenance for other cases
    else:
        return 'maintenance'


# Apply labeling
df['recommendation'] = df.apply(label_data, axis=1)

# Print data distribution after labeling
print("Training data distribution:")
print(df['recommendation'].value_counts())
print(df[['repair_count', 'average_cost']].describe())

# Define target variable
y = df['recommendation']

# Balance the dataset by upsampling minority classes
max_size = df['recommendation'].value_counts().max()
df_dispose = df[df['recommendation'] == 'dispose']
df_repair = df[df['recommendation'] == 'repair']
df_maintenance = df[df['recommendation'] == 'maintenance']

# Upsample minority classes
df_dispose_upsampled = resample(df_dispose, replace=True, n_samples=max_size, random_state=42)
df_repair_upsampled = resample(df_repair, replace=True, n_samples=max_size, random_state=42)
df_maintenance_upsampled = resample(df_maintenance, replace=True, n_samples=max_size, random_state=42)

# Combine all upsampled data
df_balanced = pd.concat([df_dispose_upsampled, df_repair_upsampled, df_maintenance_upsampled])

# Define features and target after balancing
X_balanced = df_balanced[['purchase_cost', 'lifespan_years', 'salvage_value', 'depreciation_per_year',
                          'repair_count', 'average_cost', 'time_between_repairs']]
y_balanced = df_balanced['recommendation']

# Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

# Train the model with class weighting
model = RandomForestClassifier(class_weight='balanced')  # Using Random Forest for better performance
model.fit(X_train, y_train)

# Evaluate model performance with accuracy score and confusion matrix
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
conf_matrix = confusion_matrix(y_test, predictions)
class_report = classification_report(y_test, predictions)

print(f"Model accuracy: {accuracy}")
print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", class_report)

# Save the trained model
joblib.dump(model, 'maintenance_model.pkl')
