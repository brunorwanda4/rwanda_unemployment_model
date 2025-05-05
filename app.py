# app.py
from flask import Flask, render_template, request, flash
import joblib
import pandas as pd
from pymongo import MongoClient, DESCENDING # Import DESCENDING for sorting
import datetime
import os
import numpy as np # For handling potential numeric issues

app = Flask(__name__)
# Secret key for flashing messages (optional but good practice)
app.secret_key = os.urandom(24) # Replace with a fixed secret key in production

# --- Configuration ---
MODEL_DIR = 'rwanda_unemployment_model'
MODEL_PATH = os.path.join(MODEL_DIR, 'unemployment_model.pkl')
ENCODERS_PATH = os.path.join(MODEL_DIR, 'label_encoders.pkl')

# --- MongoDB Setup (Optional) ---
# Set to None if you don't want to use MongoDB for history
MONGO_URI = 'mongodb://localhost:27017/' # Your MongoDB connection string
DB_NAME = 'rwanda_unemployment_db'
COLLECTION_NAME = 'predictions'
predictions_collection = None # Initialize as None

try:
    if MONGO_URI:
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]
        predictions_collection = db[COLLECTION_NAME]
        # The ismaster command is cheap and does not require auth.
        client.admin.command('ismaster')
        print("Successfully connected to MongoDB!")
    else:
        print("MongoDB URI not set. History feature disabled.")
except Exception as e:
    print(f"Could not connect to MongoDB: {e}")
    predictions_collection = None # Ensure it's None on connection error
# --- End MongoDB Setup ---


# --- Load model and encoders ---
try:
    model = joblib.load(MODEL_PATH)
    label_encoders = joblib.load(ENCODERS_PATH)
    print(f"Model loaded from {MODEL_PATH}")
    print(f"Encoders loaded from {ENCODERS_PATH}")
    # --- IMPORTANT: Get the feature order the model was trained on ---
    # This assumes your training script saved features in X before splitting
    # If not, you might need to load X_train.columns separately or define manually
    # For now, defining manually based on the likely order from training script:
    # (Adjust this list based on the actual columns used in X during training!)
    # In app.py - this list should be correct now
    EXPECTED_FEATURES = [
        'Sex', 'Relationship', 'Age', 'Marital_status', 'Unpaid_work',
        'Contract_duration', 'youngs', 'Educaional_level', 'age5', 'hhsize',
        'TVT2', 'unemployment_duration', 'Field_of_education', 'occupation'
        # NO 'Unnamed: 16' here!
    ]
    print(f"Model expects features in this order: {EXPECTED_FEATURES}")

except FileNotFoundError:
    print(f"Error: Model or encoder file not found.")
    print(f"Please ensure '{MODEL_PATH}' and '{ENCODERS_PATH}' exist.")
    print("Run the training script first.")
    exit()
except Exception as e:
    print(f"Error loading model or encoders: {e}")
    exit()
# --- End Load model ---

# --- Helper Function for Preprocessing ---
def preprocess_input(form_data, encoders, expected_features):
    """Prepares form data for prediction."""
    processed_data = {}
    original_data = {} # Keep original string values for saving

    # Process numeric fields first
    numeric_fields = ['Age', 'hhsize', 'unemployment_duration'] # Adjust as needed
    for field in numeric_fields:
        if field in expected_features:
            try:
                value = float(form_data.get(field, 0)) # Default to 0 if missing? Or handle error
                processed_data[field] = value
                original_data[field] = value
            except (ValueError, TypeError):
                raise ValueError(f"Invalid numeric value for '{field}': {form_data.get(field)}")

    # Process categorical fields using encoders
    categorical_fields = [f for f in expected_features if f not in numeric_fields]
    for col in categorical_fields:
         if col in encoders:
            le = encoders[col]
            value = form_data.get(col, 'Unknown') # Default to 'Unknown' if missing? Needs training alignment
            original_data[col] = value # Store original string

            # Check if the value is known to the encoder
            if value in le.classes_:
                 processed_data[col] = le.transform([value])[0]
            else:
                 # Handle unknown category: Option 1: Try 'Unknown' if it was used in training
                 if 'Unknown' in le.classes_:
                     print(f"Warning: Unknown category '{value}' for '{col}'. Using 'Unknown'.")
                     processed_data[col] = le.transform(['Unknown'])[0]
                     original_data[col] = 'Unknown' # Update original if using fallback
                 else:
                    # Option 2: Raise error if 'Unknown' wasn't trained or strict matching is needed
                    raise ValueError(f"Unknown category '{value}' for column '{col}' and 'Unknown' not trained.")
         else:
             # This shouldn't happen if expected_features aligns with encoders
             print(f"Warning: No encoder found for expected categorical feature '{col}'. Skipping.")

    # Create DataFrame in the correct order
    try:
        input_df = pd.DataFrame([processed_data])
        # Reorder columns to match the training order
        input_df = input_df[expected_features]
    except KeyError as e:
        raise ValueError(f"Missing expected feature column during DataFrame creation: {e}. Check form names and EXPECTED_FEATURES list.")

    return input_df, original_data
# --- End Helper Function ---


@app.route('/', methods=['GET', 'POST'])
def predict_unemployment():
    result_text = None
    history_data = []
    form_data = request.form if request.method == 'POST' else {} # Keep form data for re-population

    # --- Fetch History Data (if MongoDB is enabled) ---
    if predictions_collection is not None:
        try:
            history_cursor = predictions_collection.find().sort('timestamp', DESCENDING).limit(10)
            history_data = list(history_cursor)
        except Exception as e:
            print(f"Error fetching history from MongoDB: {e}")
            flash("Could not retrieve prediction history.", "warning")
    # --- End Fetch History Data ---

    if request.method == 'POST':
        try:
            # Get user input from form
            user_input_form = request.form.to_dict()

            # Preprocess the input
            input_df, user_input_original = preprocess_input(user_input_form, label_encoders, EXPECTED_FEATURES)

            # Make prediction
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0] # Get probabilities

            # Interpret prediction
            unemployed_prob = probability[1] # Assuming 1 is the 'Unemployed' class
            if prediction == 1:
                result_text = f"Prediction: Likely Unemployed 😟 (Confidence: {unemployed_prob:.2%})"
            else:
                result_text = f"Prediction: Likely Employed/Other ✅ (Confidence: {1-unemployed_prob:.2%})"

            # --- Save data to MongoDB (if enabled) ---
            if predictions_collection is not None:
                try:
                    document_to_save = user_input_original.copy()
                    document_to_save['timestamp'] = datetime.datetime.now(datetime.timezone.utc)
                    document_to_save['prediction_result_code'] = int(prediction)
                    document_to_save['prediction_result_text'] = result_text
                    document_to_save['prediction_probability_unemployed'] = float(unemployed_prob) # Store probability
                    insert_result = predictions_collection.insert_one(document_to_save)
                    print(f"Data saved to MongoDB with ID: {insert_result.inserted_id}")

                    # Refresh history after saving to show the latest entry immediately
                    history_cursor = predictions_collection.find().sort('timestamp', DESCENDING).limit(10)
                    history_data = list(history_cursor) # Update history list

                except Exception as e:
                    print(f"Error saving data to MongoDB: {e}")
                    flash("Could not save prediction to history.", "error")
            # --- End Save ---

            flash(result_text, "success") # Use flash for non-blocking message

        except ValueError as ve:
            print(f"Data processing error: {ve}")
            flash(f"Error in input data: {ve}", "danger")
            result_text = None # Clear result on error
        except KeyError as ke:
            print(f"Missing data error: {ke}")
            flash(f"Missing input field or configuration error: {ke}", "danger")
            result_text = None
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            flash("An unexpected error occurred during prediction.", "danger")
            result_text = None

        # Re-render form, potentially re-populating with submitted values
        # Pass result_text=None because we used flash for messages
        return render_template('form.html', history=history_data, form_data=form_data, result=None)

    # For GET requests, render the empty form with history
    return render_template('form.html', history=history_data, form_data={}, result=None)

# --- Function to provide options for dropdowns ---
@app.context_processor
def inject_options():
    """Makes category options available to the template."""
    options = {}
    for col, encoder in label_encoders.items():
        # Convert numpy array to list for Jinja
        options[col + '_options'] = encoder.classes_.tolist()
    return dict(options=options)
# --- End Options Function ---


if __name__ == '__main__':
    # Use debug=False in production
    # Use host='0.0.0.0' to make it accessible on your network
    app.run(debug=True, host='0.0.0.0', port=5000)