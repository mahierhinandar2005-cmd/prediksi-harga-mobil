import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Prediksi Harga Mobil", page_icon="🚗")
st.title("🚗 Prediksi Harga Mobil Listrik Indonesia 2026")

@st.cache_resource
def load():
    model = joblib.load("model_ev.pkl")
    scaler_X = joblib.load("scaler_X.pkl")
    scaler_y = joblib.load("scaler_y.pkl")
    return model, scaler_X, scaler_y

model, scaler_X, scaler_y = load()
st.success("✅ Model siap!")

st.subheader("Masukkan Spesifikasi Mobil")

col1, col2 = st.columns(2)

with col1:
    tahun = st.number_input("Tahun", 2020, 2026, 2026)
    baterai = st.number_input("Baterai (kWh)", 20.0, 150.0, 60.0)

with col2:
    jarak = st.number_input("Range (km)", 100, 800, 400)
    daya = st.number_input("Daya (hp)", 30, 600, 150)

if st.button("Prediksi Harga"):
    input_data = np.array([[tahun, baterai, jarak, daya]])
    input_scaled = scaler_X.transform(input_data)
    pred_scaled = model.predict(input_scaled)
    harga = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))
    st.success(f"💰 Estimasi Harga: Rp {harga[0][0]:,.0f} Juta")
