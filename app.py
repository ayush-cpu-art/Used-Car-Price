import pickle
import numpy as np
import pandas as pd
import streamlit as st


CURRENT_YEAR = 2021  # reference year used during training (car_age = CURRENT_YEAR - year)

ARTIFACT_FILES = {
    "model": "used_car_price_model.pkl",
    "scaler": "scaler.pkl",
    "fuel_encoder": "fuel_encoder.pkl",
    "brand_encoder": "brand_encoder.pkl",
    "feature_columns": "feature_columns.pkl",
}

DATASET_URL = "https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho"



@st.cache_resource(show_spinner="Loading model...")
def load_artifacts():
    loaded = {}
    for key, filename in ARTIFACT_FILES.items():
        with open(filename, "rb") as f:
            loaded[key] = pickle.load(f)
    return loaded


def build_feature_row(inputs: dict, encoders: dict, feature_columns: list) -> pd.DataFrame:
    """Turn raw form inputs into the exact feature row the model expects."""
    car_age = max(CURRENT_YEAR - inputs["year"], 1)
    is_automatic = int(inputs["transmission"] == "Automatic")

    row = {
        "present_price": inputs["present_price"],
        "kms_driven": inputs["kms_driven"],
        "fuel_type": encoders["fuel_encoder"].transform([inputs["fuel_type"]])[0],
        "transmission_encoded": is_automatic,
        "owner": inputs["owner"],
        "brand": encoders["brand_encoder"].transform([inputs["brand"]])[0],
        "car_age": car_age,
        "kms_per_year": round(inputs["kms_driven"] / car_age),
        "is_automatic": is_automatic,
        "is_first_owner": int(inputs["owner"] == 0),
        "is_dealer": int(inputs["seller_type"] == "Dealer"),
    }
    return pd.DataFrame([row])[feature_columns]



st.set_page_config(page_title="Used Car Price Predictor", page_icon="🚗", layout="wide")

artifacts = load_artifacts()
model = artifacts["model"]
scaler = artifacts["scaler"]
fuel_encoder = artifacts["fuel_encoder"]
brand_encoder = artifacts["brand_encoder"]
feature_columns = artifacts["feature_columns"]


with st.sidebar:
    st.header("🚗 About this app")
    st.markdown(
        "Estimates the resale value of a used car from its age, mileage, "
        "and specifications, using a tuned **XGBoost regressor**."
    )
    st.divider()
    st.markdown("**Dataset**")
    st.markdown(f"[Vehicle dataset from CarDekho]({DATASET_URL})")
    st.markdown("**Currency**")
    st.markdown("All prices in **Lakhs (₹, INR)**")
    st.divider()
    st.caption("Built with Streamlit · scikit-learn · XGBoost")


st.title("Used Car Price Prediction")
st.caption("Fill in the car's details below to get an estimated resale value.")


left, right = st.columns([1.1, 1], gap="large")

with left:
    with st.form("prediction_form"):
        st.subheader("Car details")

        c1, c2 = st.columns(2)
        with c1:
            brand = st.selectbox("Brand", sorted(brand_encoder.classes_))
            year = st.number_input(
                "Year of purchase", min_value=2003, max_value=CURRENT_YEAR, value=2017, step=1
            )
            present_price = st.number_input(
                "Present ex-showroom price (Lakhs)",
                min_value=2.5, max_value=60.0, value=8.5, step=0.1,
            )
        with c2:
            kms_driven = st.number_input(
                "Kilometers driven", min_value=500, max_value=500_000, value=45_000, step=500
            )
            fuel_type = st.selectbox("Fuel type", sorted(fuel_encoder.classes_))
            owner = st.selectbox("Previous owners", [0, 1, 2, 3], index=0)

        st.subheader("Other specs")
        c3, c4 = st.columns(2)
        with c3:
            transmission = st.radio("Transmission", ["Manual", "Automatic"], horizontal=True)
        with c4:
            seller_type = st.radio("Seller type", ["Dealer", "Individual"], horizontal=True)

        submitted = st.form_submit_button(
            "Predict price", use_container_width=True, type="primary"
        )

with right:
    st.subheader("Estimated resale value")

    if not submitted:
        st.info("Fill in the form and click **Predict price** to see an estimate here.")
    else:
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
        input_row = build_feature_row(inputs, artifacts, feature_columns)
        prediction = float(model.predict(scaler.transform(input_row))[0])
        prediction = max(prediction, 0.1)  # guard against negative predictions on edge inputs

        st.metric("Predicted selling price", f"₹ {prediction:.2f} Lakhs")

        depreciation_pct = (1 - prediction / present_price) * 100 if present_price > 0 else 0
        st.caption(f"≈ **{depreciation_pct:.0f}%** below the present ex-showroom price.")

        car_age = max(CURRENT_YEAR - year, 1)
        m1, m2, m3 = st.columns(3)
        m1.metric("Car age", f"{car_age} yr")
        m2.metric("Kms / year", f"{kms_driven // car_age:,}")
        m3.metric("Owners", owner)

        with st.expander("See model input"):
            st.dataframe(input_row.T.rename(columns={0: "Value"}), use_container_width=True)

st.divider()
st.caption(f"Dataset: [CarDekho used vehicles]({DATASET_URL}) · Model: tuned XGBoost regressor")