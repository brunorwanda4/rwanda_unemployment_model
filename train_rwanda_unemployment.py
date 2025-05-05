import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- Configuration ---
DATA_PATH = 'data/data.csv'
MODEL_DIR = 'rwanda_unemployment_model'
MODEL_PATH = os.path.join(MODEL_DIR, 'unemployment_model.pkl')
ENCODERS_PATH = os.path.join(MODEL_DIR, 'label_encoders.pkl')

# Ensure output directory exists
os.makedirs(MODEL_DIR, exist_ok=True)
print(f"Ensured model directory exists: {MODEL_DIR}")

# --- Load Data ---
print(f"Loading data from: {DATA_PATH}")
try:
    data = pd.read_csv(DATA_PATH, na_values=['', ' ', 'N/A', 'NaN', ',,'])
except FileNotFoundError:
    print(f"Error: Data file not found at {DATA_PATH}")
    exit()
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

print("Data loaded successfully. Initial shape:", data.shape)

# --- Drop unnamed columns ---
unnamed_cols = [col for col in data.columns if 'Unnamed:' in col]
if unnamed_cols:
    print(f"Found and dropping unnamed columns: {unnamed_cols}")
    data = data.drop(columns=unnamed_cols)
    print("Shape after dropping unnamed columns:", data.shape)

print("Initial missing values (first 10 columns):\n", data.isnull().sum().head(10))

# --- Create target variable ---
TARGET_COL = 'LFP'
TARGET_VALUE_UNEMPLOYED = 'Unemployed'
NEW_TARGET_COL = 'is_unemployed'

if TARGET_COL not in data.columns:
    print(f"Error: Target column '{TARGET_COL}' not found in the data.")
    exit()

data[NEW_TARGET_COL] = (data[TARGET_COL] == TARGET_VALUE_UNEMPLOYED).astype(int)
print(f"\nCreated target variable '{NEW_TARGET_COL}':")
print(data[NEW_TARGET_COL].value_counts(dropna=False))

# --- Drop unnecessary columns ---
cols_to_drop = [TARGET_COL]
if 'pid' in data.columns:
    cols_to_drop.append('pid')

features = data.drop(columns=cols_to_drop + [NEW_TARGET_COL]).columns.tolist()
print(f"\nUsing features: {features}")

X = data[features].copy()
y = data[NEW_TARGET_COL]

print("\nVerifying features used for training:")
print(X.columns.tolist())

# --- Encode categorical features ---
label_encoders = {}
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
print("Categorical columns to encode:", categorical_cols)

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = X[col].astype(str).fillna("Missing")  # Convert all to string + handle NaNs
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# --- Handle numeric missing values ---
X = X.fillna(0)

# --- Train/Test Split ---
print("\nSplitting data into training and testing sets...")
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set size: {X_train.shape[0]} samples ({y_train.mean()*100:.2f}% unemployed)")
    print(f"Testing set size: {X_test.shape[0]} samples ({y_test.mean()*100:.2f}% unemployed)")
except ValueError as e:
    print(f"Error during train/test split: {e}")
    exit()

# --- Train Model ---
print("\nTraining RandomForestClassifier model...")
model = RandomForestClassifier(random_state=42, class_weight='balanced', n_estimators=100)
model.fit(X_train, y_train)
print("Model training complete.")

# --- Evaluation ---
print("\n--- Model Evaluation ---")
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

print("\nTraining Set Performance:")
print(f"Accuracy: {accuracy_score(y_train, y_pred_train):.4f}")

print("\nTesting Set Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_test):.4f}")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_test))
print("\nClassification Report:\n", classification_report(
    y_test, y_pred_test, target_names=['Employed/Other', 'Unemployed'], zero_division=0
))
print("------------------------\n")

# --- Save Model and Encoders ---
print(f"Saving model to: {MODEL_PATH}")
joblib.dump(model, MODEL_PATH)

print(f"Saving label encoders to: {ENCODERS_PATH}")
joblib.dump(label_encoders, ENCODERS_PATH)

print("\nModel and encoders saved successfully!")
