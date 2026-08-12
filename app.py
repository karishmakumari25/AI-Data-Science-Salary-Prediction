import streamlit as st
import pandas as pd
import joblib
from pandas.api.types import is_numeric_dtype

st.set_page_config(
    page_title="AI Salary Prediction",
    page_icon="💰",
    layout="wide"
)

df = pd.read_csv("dataset/ai_ds_job_salaries_2026.csv")
model = joblib.load("models/salary_prediction_model.pkl")


st.title("💰 AI Salary Prediction")
st.write("Predict your expected salary using Machine Learning.")

st.divider()

st.subheader("Enter Job Details")

input_data = {}

# Remove target column
features = df.drop("salary_usd", axis=1).columns

col1, col2 = st.columns(2)

for i, column in enumerate(features):

    # Numeric columns
    if is_numeric_dtype(df[column]):

        minimum = float(df[column].min())
        maximum = float(df[column].max())
        default = float(df[column].median())

        with col1 if i % 2 == 0 else col2:
            input_data[column] = st.number_input(
                column.replace("_", " ").title(),
                min_value=minimum,
                max_value=maximum,
                value=default
            )

    # Text / categorical columns
    else:

        options = sorted(
            df[column].dropna().astype(str).unique().tolist()
        )

        with col1 if i % 2 == 0 else col2:
            input_data[column] = st.selectbox(
                column.replace("_", " ").title(),
                options
            )

st.divider()

# Prediction button
if st.button("🔮 Predict Salary", use_container_width=True):

    input_df = pd.DataFrame([input_data])

    prediction = model.predict(input_df)[0]

    st.success(
        f"💰 Predicted Salary: ${prediction:,.2f} per year"
    )

    st.info(
        "Prediction generated using the trained Linear Regression model."
    )