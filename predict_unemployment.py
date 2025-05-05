import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib # For saving the model and preprocessor
import os # To handle file paths

# --- 1. Load and Prepare Data ---
# Define the path to the dataset
# IMPORTANT: Ensure this path is correct relative to where the script is run.
# If 'data' is a folder in the same directory as the script:
data_file_path = 'data/data.csv'
# If the path is absolute, use the full path, e.g., '/path/to/your/data/data.csv'

try:
    # Load the dataset from the CSV file
    df = pd.read_csv(data_file_path)
    print(f"Successfully loaded data from {data_file_path}")
    print("Initial data shape:", df.shape)
    print("Columns:", df.columns.tolist())

except FileNotFoundError:
    print(f"Error: The file '{data_file_path}' was not found.")
    print("Please ensure the CSV file exists at the specified location.")
    # Exit or handle the error appropriately if the file is essential
    exit() # Exit the script if data loading fails
except Exception as e:
    print(f"An error occurred while loading the data: {e}")
    exit()

# --- Data Cleaning/Consistency (Example) ---
# It's common for CSV data to have leading/trailing spaces, especially in string columns
# Example: Clean the 'hhsize' column if it has inconsistent spacing like ' 9 persons' vs '4 Persons'
if 'hhsize' in df.columns:
    df['hhsize'] = df['hhsize'].str.strip() # Remove leading/trailing whitespace

# --- Define Youth ---
# Typically 15-35 years old. Adjust as needed for Rwanda context.
# Ensure 'Age' column exists and is numeric
if 'Age' not in df.columns:
    print("Error: 'Age' column not found in the dataset.")
    exit()
# Convert Age to numeric, coercing errors to NaN (which will be handled later or dropped)
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
df.dropna(subset=['Age'], inplace=True) # Drop rows where Age could not be converted
df['Age'] = df['Age'].astype(int)

df_youth = df[(df['Age'] >= 15) & (df['Age'] <= 35)].copy()
print("Youth data shape (Age 15-35):", df_youth.shape)

if df_youth.empty:
    print("Warning: No data found for the youth age range (15-35). Check your data and age filter.")
    exit()

# --- Define Target Variable ---
# Target: 1 if Unemployed, 0 otherwise (Employed or Inactive)
# Ensure 'LFP' (Labor Force Participation) column exists
if 'LFP' not in df.columns:
    print("Error: 'LFP' column (Labor Force Participation) not found. This is needed to define the target variable.")
    exit()

# We consider only those defined as 'Unemployed' in LFP as the target class 1.
# Handle potential variations in the 'Unemployed' string (e.g., case sensitivity)
df_youth['Is_Unemployed'] = df_youth['LFP'].str.strip().str.lower().apply(lambda x: 1 if x == 'unemployed' else 0)

# Drop rows where LFP is missing for the youth subset, as it's our target basis
df_youth.dropna(subset=['LFP'], inplace=True)
print("Youth data shape after dropping missing LFP:", df_youth.shape)

if df_youth.empty:
    print("Warning: No youth data remaining after handling missing LFP values.")
    exit()

# --- 2. Feature Selection ---
# Select features potentially relevant for predicting unemployment
# Ensure these columns exist in your CSV
potential_features = ['Sex', 'Age', 'Marital_status', 'Educaional_level', 'hhsize', 'TVT2', 'Field_of_education', 'Relationship']

# Check which potential features are actually present in the loaded dataframe
features = [col for col in potential_features if col in df_youth.columns]
missing_potential_features = [col for col in potential_features if col not in df_youth.columns]

if not features:
     print("Error: None of the selected features were found in the dataset. Check column names.")
     exit()
if missing_potential_features:
    print(f"Warning: The following potential features were not found in the dataset and will be excluded: {missing_potential_features}")

print(f"Using features: {features}")

X = df_youth[features]
y = df_youth['Is_Unemployed']

# --- 3. Preprocessing ---
# Identify categorical and numerical features *within the selected features X*
categorical_features = X.select_dtypes(include=['object', 'category']).columns
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns # Should include 'Age'

print(f"Numerical features for preprocessing: {numerical_features.tolist()}")
print(f"Categorical features for preprocessing: {categorical_features.tolist()}")

# Create preprocessing pipelines for numerical and categorical features
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')) # Impute missing numerical values with median
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')), # Impute missing categorical values with most frequent
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) # Convert categories to numbers, handle unknown categories during prediction
])

# Create a preprocessor object using ColumnTransformer
# Only include transformers if there are features of that type
transformers_list = []
if not numerical_features.empty:
    transformers_list.append(('num', numerical_transformer, numerical_features))
if not categorical_features.empty:
    transformers_list.append(('cat', categorical_transformer, categorical_features))

if not transformers_list:
    print("Error: No numerical or categorical features identified for preprocessing.")
    exit()

preprocessor = ColumnTransformer(
    transformers=transformers_list,
    remainder='passthrough' # Keep other columns if any (shouldn't be any based on selection)
)

# --- 4. Model Definition ---
# Using RandomForestClassifier as it handles non-linearities well
# Using class_weight='balanced' helps if unemployment is rare (imbalanced dataset)
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', RandomForestClassifier(n_estimators=100,
                                                               random_state=42,
                                                               class_weight='balanced',
                                                               n_jobs=-1))]) # Use all available CPU cores

# --- 5. Train the Model ---
# Split data, stratify by y to maintain class proportions in train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

print(f"Training data shape: X_train {X_train.shape}, y_train {y_train.shape}")
print(f"Testing data shape: X_test {X_test.shape}, y_test {y_test.shape}")

# Check if training set is empty
if X_train.empty or y_train.empty:
    print("Error: Training data is empty after splitting. Check data processing steps.")
    exit()

print("Starting model training...")
model.fit(X_train, y_train)
print("Model training complete.")

# --- 6. Evaluate the Model ---
# Check if test set is empty
if X_test.empty or y_test.empty:
    print("Warning: Test data is empty. Evaluation cannot be performed.")
else:
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nModel Accuracy on Test Set: {accuracy:.4f}")
    # Note: Achieving >98% accuracy is highly dependent on data quality, feature relevance,
    # model complexity, and the inherent predictability of the problem. It's often unrealistic.
    # Focus on other metrics like precision, recall, F1-score, especially for imbalanced datasets.
    print("\nClassification Report:")
    # Added target_names for better readability if classes are 0 and 1
    print(classification_report(y_test, y_pred, zero_division=0, target_names=['Not Unemployed (0)', 'Unemployed (1)']))
    print("\nConfusion Matrix:")
    # Display with labels for clarity
    cm = confusion_matrix(y_test, y_pred)
    print(pd.DataFrame(cm, index=['Actual Not Unemployed', 'Actual Unemployed'], columns=['Predicted Not Unemployed', 'Predicted Unemployed']))


# --- 7. Save the Model and Preprocessor ---
# In a real application, save the trained model pipeline to disk
# This saves both the preprocessor and the classifier
# model_save_path = 'unemployment_model_pipeline.joblib'
# try:
#     joblib.dump(model, model_save_path)
#     print(f"\nModel pipeline saved successfully to {model_save_path}")
# except Exception as e:
#     print(f"\nError saving the model pipeline: {e}")
print("\nModel training and evaluation complete. (Model not saved in this example)")


# --- 8. Prediction Function ---
# Function to predict unemployment for new data
def predict_unemployment(new_data_dict):
    """
    Predicts unemployment status for new individual data using the trained pipeline.

    Args:
        new_data_dict (dict): A dictionary containing the features for one individual.
                              Keys must match the feature names used during training.
                              Example: {'Sex': 'Female', 'Age': 22, ...}

    Returns:
        tuple: (prediction, probability)
               prediction (int): 1 if predicted unemployed, 0 otherwise.
               probability (float): Probability of the positive class (unemployed).
               Returns (None, None) if prediction fails.
    """
    try:
        # Ensure the input dictionary keys match the required features
        if not all(f in new_data_dict for f in features):
            missing_keys = [f for f in features if f not in new_data_dict]
            print(f"Error: Missing features in input dictionary: {missing_keys}")
            return None, None

        # Convert dict to DataFrame - IMPORTANT: ensure columns match training features order/names
        new_data_df = pd.DataFrame([new_data_dict], columns=features)

        # Load the saved model pipeline (in a real app)
        # loaded_model = joblib.load('unemployment_model_pipeline.joblib')
        # Use the loaded model:
        # prediction = loaded_model.predict(new_data_df)
        # probabilities = loaded_model.predict_proba(new_data_df)

        # Use the trained model pipeline directly (as it's in memory here)
        prediction = model.predict(new_data_df)
        probabilities = model.predict_proba(new_data_df) # Gets probabilities for all classes

        predicted_class = prediction[0]
        probability_unemployed = probabilities[0][1] # Probability of the positive class (1)

        # print(f"Input: {new_data_dict}")
        # print(f"Predicted class: {predicted_class}")
        # print(f"Prediction probabilities (Not Unemployed, Unemployed): {probabilities[0]}")

        return predicted_class, probability_unemployed

    except Exception as e:
        print(f"Error during prediction: {e}")
        return None, None

# --- Example Prediction ---
# Example of how to use the prediction function
# Ensure the example data uses values consistent with the training data format
# (e.g., 'hhsize' format, feature names)
example_individual = {
    'Sex': 'Male',
    'Age': 25,
    'Marital_status': 'Single',
    'Educaional_level': 'Lower secondary',
    'hhsize': '4 Persons', # Match potential format in data
    'TVT2': 'Completed general',
    'Field_of_education': 'General education',
    'Relationship': 'Child (Son/daughter)'
}

# Filter example_individual to only include features used in the model
example_individual_filtered = {k: v for k, v in example_individual.items() if k in features}

print("\n--- Example Prediction ---")
predicted_status, probability = predict_unemployment(example_individual_filtered)

if predicted_status is not None:
    status_label = "Unemployed" if predicted_status == 1 else "Not Unemployed (Employed/Inactive)"
    print(f"Prediction for example individual: {status_label} (Probability: {probability:.2f})")
else:
    print("Prediction failed for the example individual.")


# --- 9. Database Connection (Placeholder) ---
# Placeholder for MySQL connection logic
# import mysql.connector
#
# db_config = {
#     'host': "your_mysql_host",
#     'user': "your_username",
#     'password': "your_password",
#     'database': "your_database_name"
# }
#
# def store_prediction(data_dict, prediction, probability):
#     connection = None
#     cursor = None
#     try:
#         connection = mysql.connector.connect(**db_config)
#         cursor = connection.cursor()
#         print("MySQL Database connection successful")
#
#         # Adapt the query and values based on your table structure
#         # Ensure column names in the table match the keys in data_dict + prediction/probability
#         query = """INSERT INTO predictions
#                    (Sex, Age, Marital_status, Educaional_level, hhsize, TVT2, Field_of_education, Relationship, predicted_status, probability)
#                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
#         values = (
#             data_dict.get('Sex'), data_dict.get('Age'), data_dict.get('Marital_status'),
#             data_dict.get('Educaional_level'), data_dict.get('hhsize'), data_dict.get('TVT2'),
#             data_dict.get('Field_of_education'), data_dict.get('Relationship'),
#             int(prediction), float(probability) # Ensure correct types for DB
#         )
#
#         cursor.execute(query, values)
#         connection.commit()
#         print(f"Prediction for Age {data_dict.get('Age')} saved to database.")
#
#     except mysql.connector.Error as err:
#         print(f"Error connecting to or writing to MySQL: {err}")
#     finally:
#         if cursor:
#             cursor.close()
#         if connection and connection.is_connected():
#             connection.close()
#             # print("MySQL connection is closed")

# Example of storing the prediction result
# if predicted_status is not None:
#     store_prediction(example_individual_filtered, predicted_status, probability)

