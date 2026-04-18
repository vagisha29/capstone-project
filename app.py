import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier


st.title("🌱 Crop Recommendation System")
st.write("Enter soil and environmental conditions to get the best crop.")


uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

   
    if "Unnamed: 0" in df.columns:
        df = df.drop("Unnamed: 0", axis=1)
    elif "H1" in df.columns:
        df = df.drop("H1", axis=1)

    
    le_soil = LabelEncoder()
    le_crop = LabelEncoder()

    df["Soil"] = le_soil.fit_transform(df["Soil"])
    df["Crop"] = le_crop.fit_transform(df["Crop"])

    
    X = df[[
        "Temperature", "Humidity", "Rainfall", "PH",
        "Nitrogen", "Phosphorous", "Potassium", "Carbon", "Soil"
    ]]
    y = df["Crop"]

    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    st.success("Model trained successfully ✅")

    # ---------------- INPUT UI ---------------- #

    st.subheader("Enter Values")

    col1, col2 = st.columns(2)

    with col1:
        temperature = st.number_input("Temperature (°C)")
        humidity = st.number_input("Humidity (%)")
        rainfall = st.number_input("Rainfall (mm)")
        ph = st.number_input("pH (0–14)", min_value=0.0, max_value=14.0)

    with col2:
        nitrogen = st.number_input("Nitrogen (kg/ha)", min_value=0.0)
        phosphorous = st.number_input("Phosphorous (kg/ha)", min_value=0.0)
        potassium = st.number_input("Potassium (kg/ha)", min_value=0.0)
        carbon = st.number_input("Carbon (%)", min_value=0.0)

    
    soil_options = list(le_soil.classes_)
    soil = st.selectbox("Soil Type", soil_options)

    soil_value = le_soil.transform([soil])[0]

    # ---------------- PREDICTION ---------------- #

    if st.button("Predict Crop 🌾"):

        new_data = pd.DataFrame([[
            temperature, humidity, rainfall, ph,
            nitrogen, phosphorous, potassium, carbon, soil_value
        ]], columns=X.columns)

        prediction = model.predict(new_data)[0]
        crop_name = le_crop.inverse_transform([prediction])[0]

        st.success(f"🌱 Recommended Crop: **{crop_name}**")