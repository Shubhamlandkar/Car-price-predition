import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score

# --- App Title ---
st.title("🚗 Car Price Prediction App")
st.write("Predict your car's price based on its features using Machine Learning!")

# --- Load or Create Dataset ---
st.sidebar.header("Dataset Options")
use_example = st.sidebar.checkbox("Use Example Dataset", value=True)

if use_example:
    # Simple sample dataset (you can replace with your own CSV)
    data = {
        "Company": ["Toyota", "BMW", "Hyundai", "Audi", "Mercedes", "Honda", "Tata", "Ford", "Hyundai", "Toyota"],
        "Year": [2015, 2018, 2013, 2019, 2017, 2014, 2020, 2016, 2019, 2021],
        "Fuel_Type": ["Petrol", "Diesel", "Petrol", "Diesel", "Petrol", "Diesel", "Petrol", "Diesel", "Petrol", "Diesel"],
        "Kms_Driven": [50000, 30000, 70000, 20000, 40000, 60000, 15000, 45000, 25000, 10000],
        "Price": [500000, 800000, 350000, 1200000, 900000, 450000, 700000, 550000, 800000, 950000]
    }
    df = pd.DataFrame(data)
else:
    uploaded_file = st.sidebar.file_uploader("Upload your car dataset (CSV)", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
    else:
        st.warning("Please upload a dataset or use example data.")
        st.stop()

# --- Data Preview ---
st.subheader("Dataset Preview")
st.dataframe(df.head())

# --- Preprocess Data ---
df = df.copy()
label_encoders = {}

for col in ["Company", "Fuel_Type"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Features and target
X = df.drop("Price", axis=1)
y = df["Price"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
st.write(f"### Model R² Score: {r2:.2f}")

# --- User Input Section ---
st.markdown("---")
st.header("🔢 Enter Car Details to Predict Price")

# Get options dynamically from data
companies = label_encoders["Company"].classes_
fuels = label_encoders["Fuel_Type"].classes_

company = st.selectbox("Car Brand", companies)
year = st.slider("Year of Manufacture", 2000, 2024, 2018)
fuel = st.selectbox("Fuel Type", fuels)
kms = st.number_input("KMs Driven", min_value=0, step=1000, value=20000)

if st.button("Predict Price"):
    # Encode categorical inputs
    company_enc = label_encoders["Company"].transform([company])[0]
    fuel_enc = label_encoders["Fuel_Type"].transform([fuel])[0]
    input_data = np.array([[company_enc, year, fuel_enc, kms]])

    predicted_price = model.predict(input_data)[0]
    st.success(f"💰 Estimated Car Price: ₹{predicted_price:,.0f}")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit and Scikit-Learn.")
