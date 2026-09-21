import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st

CURRENT_YEAR = 2021

ARTIFACT_FILES = {
    "model": "used_car_price_model.pkl",
    "scaler": "scaler.pkl",
    "fuel_encoder": "fuel_encoder.pkl",
    "brand_encoder": "brand_encoder.pkl",
    "feature_columns": "feature_columns.pkl",
}

DATASET_URL = "https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho"


@st.cache_resource
def load_artifacts():
    loaded = {}

    missing_files = [
        file for file in ARTIFACT_FILES.values()
        if not os.path.exists(file)
    ]

    if missing_files:
        st.error(
            f"Missing files: {', '.join(missing_files)}"
        )
        st.stop()

    for key, filename in ARTIFACT_FILES.items():
        with open(filename, "rb") as f:
            loaded[key] = pickle.load(f)

    return loaded


def build_feature_row(inputs, fuel_encoder, brand_encoder, feature_columns):
    car_age = max(CURRENT_YEAR - inputs["year"], 1)

    row = {
        "present_price": inputs["present_price"],
        "kms_driven": inputs["kms_driven"],
        "fuel_type": fuel_encoder.transform([inputs["fuel_type"]])[0],
        "transmission_encoded": int(inputs["transmission"] == "Automatic"),
        "owner": inputs["owner"],
        "brand": brand_encoder.transform([inputs["brand"]])[0],
        "car_age": car_age,
        "kms_per_year": round(inputs["kms_driven"] / car_age),
        "is_automatic": int(inputs["transmission"] == "Automatic"),
        "is_first_owner": int(inputs["owner"] == 0),
        "is_dealer": int(inputs["seller_type"] == "Dealer"),
    }

    df = pd.DataFrame([row])

    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    return df[feature_columns]


st.set_page_config(
    page_title="Used Car Price Predictor",
    page_icon="🚗",
    layout="wide",
)

st.title("🚗 Used Car Price Prediction")

artifacts = load_artifacts()

model = artifacts["model"]
scaler = artifacts["scaler"]
fuel_encoder = artifacts["fuel_encoder"]
brand_encoder = artifacts["brand_encoder"]
feature_columns = artifacts["feature_columns"]

with st.sidebar:
    st.header("About")
    st.write("Predict the resale value of a used car.")
    st.markdown(f"[Dataset]({DATASET_URL})")

st.subheader("Enter Car Details")

col1, col2 = st.columns(2)

with col1:
    brand = st.selectbox(
        "Brand",
        sorted(list(brand_encoder.classes_))
    )

    year = st.number_input(
        "Year",
        min_value=2003,
        max_value=CURRENT_YEAR,
        value=2018,
    )

    present_price = st.number_input(
        "Present Price (Lakhs)",
        min_value=0.1,
        value=5.0,
    )

    kms_driven = st.number_input(
        "Kilometers Driven",
        min_value=0,
        value=30000,
    )

with col2:
    fuel_type = st.selectbox(
        "Fuel Type",
        sorted(list(fuel_encoder.classes_))
    )

    owner = st.selectbox(
        "Previous Owners",
        [0, 1, 2, 3]
    )

    transmission = st.selectbox(
        "Transmission",
        ["Manual", "Automatic"]
    )

    seller_type = st.selectbox(
        "Seller Type",
        ["Dealer", "Individual"]
    )

if st.button("Predict Price"):

    inputs = {
        "brand": brand,
        "year": year,
        "present_price": present_price,
        "kms_driven": kms_driven,
        "fuel_type": fuel_type,
        "owner": owner,
        "transmission": transmission,
        "seller_type": seller_type,
    }

    input_df = build_feature_row(
        inputs,
        fuel_encoder,
        brand_encoder,
        feature_columns
    )

    scaled_input = scaler.transform(input_df)

    prediction = model.predict(scaled_input)[0]

    prediction = max(prediction, 0.1)

    st.success(
        f"Estimated Selling Price: ₹ {prediction:.2f} Lakhs"
    )

    st.dataframe(input_df)