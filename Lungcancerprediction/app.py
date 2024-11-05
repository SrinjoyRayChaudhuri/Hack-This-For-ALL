import streamlit as st
import pandas as pd
import joblib

# Load the model and scaler
model = joblib.load('best_logistic_model.pkl')
scaler = joblib.load('scaler.pkl')

# Load the data
data = pd.read_csv('lung cancer data set.csv.csv')  # Update with actual file name if needed

# Ensure all columns except 'Level' are numeric
data.iloc[:, :-1] = data.iloc[:, :-1].apply(pd.to_numeric, errors='coerce')

# Calculate mean values for defaults, filling NaN with 0 as needed
feature_means = data.drop(columns=['Level', 'Patient Id', 'index'], errors='ignore').mean().fillna(0)

# Streamlit App
st.title("Lung Cancer Prediction")

# Input form
user_input = {}
with st.form("user_input_form"):
    st.subheader("Please enter your details:")
    for feature, default_val in feature_means.items():
        # Use 0 if mean value is NaN
        user_input[feature] = st.number_input(feature, value=int(default_val))

    # Submit button
    submitted = st.form_submit_button("Submit")

# Prediction logic
if submitted:
    # Convert user input to DataFrame
    input_df = pd.DataFrame([user_input])

    # Drop non-predictive columns
    input_df = input_df.drop(columns=['Patient Id', 'index'], errors='ignore')

    # Scale the input
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)

    # Output results
    st.subheader("Prediction Results:")
    if prediction[0] == 1:
        st.write("The model predicts that there is a risk of lung cancer.")
    else:
        st.write("The model predicts that there is no significant risk of lung cancer.")

    st.write(f"Confidence: {prediction_proba[0][prediction[0]] * 100:.2f}%")
