import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# LOAD MODEL & SCALER
# -----------------------------
model = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🏠 House Price Prediction")

# -----------------------------
# USER INPUT
# -----------------------------
date_val = st.date_input("Select Date")

zhvi_px = st.number_input("ZHVI Price", value=425600)
zhvi_idx = st.number_input("ZHVI Index", value=0.97)

SqFtTotLiving = st.number_input("Total Living Area (SqFt)", value=2060)
SqFtFinBasement = st.number_input("Finished Basement SqFt", value=900)

Bathrooms = st.number_input("Bathrooms", value=1.75)
Bedrooms = st.number_input("Bedrooms", value=4)
BldgGrade = st.number_input("Building Grade", value=8)

LandVal = st.number_input("Land Value", value=183000)
ImpsVal = st.number_input("Improvements Value", value=275000)

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("Predict SalePrice"):

    try:
        # Extract year & month
        year = date_val.year
        month = date_val.month

        # Create ym EXACTLY like training
        ym = year * 100 + month

        # EXACT SAME FEATURE ORDER USED IN TRAINING
        cols = [
            'ym','zhvi_px','zhvi_idx','SqFtTotLiving','SqFtFinBasement',
            'Bathrooms','Bedrooms','BldgGrade','LandVal','ImpsVal',
            'year','month'
        ]

        input_df = pd.DataFrame([[ 
            ym, zhvi_px, zhvi_idx, SqFtTotLiving, SqFtFinBasement,
            Bathrooms, Bedrooms, BldgGrade, LandVal, ImpsVal,
            year, month
        ]], columns=cols)

        # Scale
        scaled_input = scaler.transform(input_df)

        # Predict
        prediction = model.predict(scaled_input)[0]

        st.success(f"💰 Predicted SalePrice: ₹ {prediction:,.2f}")

    except Exception as e:
        st.error(f"Error: {e}")
