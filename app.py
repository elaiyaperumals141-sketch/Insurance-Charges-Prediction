
import streamlit as st
import pandas as pd
import joblib

# Load trained model and preprocessing objects
model_data = joblib.load("model.pkl")

model = model_data["model"]
scaler = model_data["scaler"]
model_columns = model_data["columns"]
scale_columns = model_data["scale_columns"]

# Page configuration
st.set_page_config(
    page_title="Insurance Charges Predictor",
    page_icon="🏥",
    layout="centered"
)

# Application title
st.title("🏥 Insurance Charges Prediction")
st.subheader("Linear Regression Machine Learning Model")

st.write(
    "Enter customer details below to predict "
    "the estimated medical insurance charges."
)

st.divider()

# Input section
st.header("Customer Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=35
    )

    sex = st.selectbox(
        "Gender",
        ["male", "female"]
    )

    bmi = st.number_input(
        "BMI",
        min_value=10.0,
        max_value=60.0,
        value=27.5,
        step=0.1
    )

with col2:
    children = st.number_input(
        "Number of Children",
        min_value=0,
        max_value=10,
        value=2
    )

    smoker = st.selectbox(
        "Smoker",
        ["no", "yes"]
    )

    region = st.selectbox(
        "Region",
        ["southeast", "southwest",
         "northeast", "northwest"]
    )

st.divider()

# Prediction button
if st.button("🔮 Predict Insurance Charges", use_container_width=True):

    # Create customer dataframe
    new_customer = pd.DataFrame({
        "age": [age],
        "sex": [sex],
        "bmi": [bmi],
        "children": [children],
        "smoker": [smoker],
        "region": [region]
    })

    # One-hot encoding
    customer_encoded = pd.get_dummies(
        new_customer,
        columns=["sex", "smoker", "region"],
        dtype=int
    )

    # Match training columns
    customer_encoded = customer_encoded.reindex(
        columns=model_columns,
        fill_value=0
    )

    # Apply the original scaler
    customer_encoded[scale_columns] = scaler.transform(
        customer_encoded[scale_columns]
    )

    # Predict charges
    prediction = model.predict(customer_encoded)[0]

    # Display result
    st.success("Prediction Completed Successfully!")

    st.subheader("Estimated Insurance Charges")

    st.metric(
        label="Predicted Charges",
        value=f"₹ {prediction:,.2f}"
    )

    st.info(
        "This is an estimated prediction generated "
        "by your Linear Regression machine learning model. "
        "Actual insurance costs may differ."
    )

st.divider()

st.caption(
    "Developed using Python, Streamlit and Scikit-learn"
)
