import streamlit as st
import pandas as pd
import joblib
import os # To build the path reliably

# --- Configuration ---
MODEL_DIR = os.path.dirname(__file__) # Assumes model is in the same directory as app.py
MODEL_PATH = os.path.join(MODEL_DIR, 'unemployment_model_pipeline.joblib')
# Define the features the model was trained on (must match predict_unemployment.py)
# You might want to load this list from a file or define it explicitly
# Ensure the order and names are EXACTLY the same as used in training!
FEATURES = ['Sex', 'Age', 'Marital_status', 'Educaional_level', 'hhsize', 'TVT2', 'Field_of_education', 'Relationship'] # Get this list reliably!


# --- Load Model ---
@st.cache_resource # Cache the model loading
def load_model(path):
    try:
        model_pipeline = joblib.load(path)
        print("Model loaded successfully.")
        return model_pipeline
    except FileNotFoundError:
        st.error(f"Error: Model file not found at {path}. Train and save the model first.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model_pipeline = load_model(MODEL_PATH)

# --- Prediction Function ---
def predict(data, model):
    if model is None:
        return None, None
    try:
        # Create DataFrame with correct column order
        df = pd.DataFrame([data], columns=FEATURES)
        prediction = model.predict(df)
        probability = model.predict_proba(df)
        return prediction[0], probability[0][1] # Return prediction and proba of class 1
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None, None

# --- Streamlit UI ---
st.set_page_config(page_title="Rwanda Youth Unemployment Prediction", layout="wide")
st.title("🇷🇼 Rwanda Youth Unemployment Prediction Dashboard")
st.markdown("""
    Enter individual details to predict the likelihood of unemployment.
    This tool uses a machine learning model trained on historical data.
    *(Note: This is a prototype based on available data)*
""")

if model_pipeline: # Only show inputs if model loaded
    st.sidebar.header("Enter Individual Data:")

    # Create input fields in the sidebar - Use options derived from your dataset analysis!
    # Example Inputs (Replace with actual categories from your data)
    sex = st.sidebar.selectbox("Sex", options=['Male', 'Female']) # Get options from df['Sex'].unique()
    age = st.sidebar.number_input("Age", min_value=15, max_value=35, value=25, step=1)
    marital_status = st.sidebar.selectbox("Marital Status", options=['Single', 'Married', 'Divorced/Separated', 'Widowed']) # Replace with actual
    education = st.sidebar.selectbox("Educational Level", options=['Never attended', 'Primary', 'Lower secondary', 'Upper secondary', 'University', 'Post-graduate']) # Replace with actual
    hhsize = st.sidebar.text_input("Household Size (e.g., '4 Persons')", value='4 Persons') # Match training data format
    tvt = st.sidebar.selectbox("TVET Status", options=['Did not attend TVT', 'Completed general', 'Completed TVT', 'Ongoing TVT']) # Replace with actual
    field_edu = st.sidebar.selectbox("Field of Education", options=['General education', 'Teacher training', 'Fine arts', 'Humanities', 'Social and behavioural science', 'Business and administration', 'Law', 'Life science', 'Physical sciences', 'Mathematics and statistics', 'Computing', 'Engineering and engineering trades', 'Manufacturing and processing', 'Architecture and building', 'Agriculture, forestry and fishery', 'Veterinary', 'Health', 'Social services', 'Personal services', 'Transport services', 'Environmental protection', 'Security services']) # Replace with actual - THIS LIST NEEDS TO BE ACCURATE
    relationship = st.sidebar.selectbox("Relationship to Head", options=['Head', 'Spouse', 'Child (Son/daughter)', 'Other relative', 'Non-relative']) # Replace with actual

    # --- Create Input Dictionary ---
    input_data = {
        'Sex': sex,
        'Age': age,
        'Marital_status': marital_status,
        'Educaional_level': education, # Ensure key matches training
        'hhsize': hhsize,
        'TVT2': tvt, # Ensure key matches training
        'Field_of_education': field_edu, # Ensure key matches training
        'Relationship': relationship # Ensure key matches training
    }

    # --- Predict Button and Display ---
    if st.sidebar.button("Predict Unemployment Status"):
        prediction, probability = predict(input_data, model_pipeline)

        st.subheader("Prediction Result")
        if prediction is not None:
            if prediction == 1:
                st.warning(f"Predicted Status: **Unemployed** (Probability: {probability:.2f})")
            else:
                st.success(f"Predicted Status: **Not Unemployed (Employed/Inactive)** (Probability of being Unemployed: {probability:.2f})")

            # --- Placeholder for Recommendations ---
            st.subheader("Potential Intervention Focus Areas (Example)")
            st.markdown("""
                *Based on the model's general findings (feature importances) and this individual's profile, consider:*
                * **If Education Level is low:** Recommend specific skills development or TVET programs relevant to market demand.
                * **If Field of Education has poor outcomes:** Suggest exploring adjacent fields or complementary skills.
                * **General:** Connect with job matching services, entrepreneurship support, etc.
                *(Tailored recommendations require deeper analysis of feature importances and intervention mapping)*
            """)
        else:
            st.error("Prediction failed. Please check inputs or model loading.")

    # --- Add Visualizations (Placeholder) ---
    st.subheader("Overall Trends (Example Visualizations)")
    st.markdown("*(Here you would add charts based on the overall dataset or aggregated predictions, e.g., unemployment rate by education level, region, etc. using Matplotlib/Seaborn/Plotly)*")
    # Example: fig, ax = plt.subplots() ... st.pyplot(fig)

else:
    st.warning("Model could not be loaded. Dashboard functionality is limited.")