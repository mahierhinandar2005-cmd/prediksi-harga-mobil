import streamlit as st
import numpy as np
import pandas as pd
import pickle

st.set_page_config(page_title="Prediksi Harga Mobil", page_icon="🚗")

st.title("🚗 Prediksi Harga Mobil Listrik Indonesia 2026")

# Load model dan scaler dengan pickle
@st.cache_resource
def load():
    # Load model
    with open("model_ev.h5", "rb") as f:
        model = pickle.load(f)
    # Load scalers
    with open("scaler_X.pkl", "rb") as f:
        scaler_X = pickle.load(f)
    with open("scaler_y.pkl", "rb") as f:
        scaler_y = pickle.load(f)
    return model, scaler_X, scaler_y

try:
    model, scaler_X, scaler_y = load()
    st.success("✅ Model berhasil dimuat!")
except Exception as e:
    st.error(f"⚠️ Gagal memuat model: {e}")
    st.info("Pastikan file model_ev.h5, scaler_X.pkl, dan scaler_y.pkl sudah ada di repository.")
    st.stop()

# Input form
st.subheader("📝 Masukkan Spesifikasi Mobil")

col1, col2 = st.columns(2)

with col1:
    tahun = st.number_input("📅 Tahun", 2020, 2026, 2026)
    baterai = st.number_input("🔋 Baterai (kWh)", 20.0, 150.0, 60.0, step=5.0)

with col2:
    jarak = st.number_input("🛣️ Range (km)", 100, 800, 400, step=25)
    daya = st.number_input("⚡ Daya (hp)", 30, 600, 150, step=10)

if st.button("💰 PREDIKSI HARGA", type="primary"):
    input_data = np.array([[tahun, baterai, jarak, daya]])
    input_scaled = scaler_X.transform(input_data)
    pred_scaled = model.predict(input_scaled)
    harga = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))
    
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.success(f"### 🎯 Estimasi Harga")
        st.metric(
            label="Harga Mobil Listrik",
            value=f"Rp {harga[0][0]:,.0f} Juta",
            delta=f"{harga[0][0]/1000:.1f} Miliar" if harga[0][0] >= 1000 else None
        )
    st.caption("⚠️ Harga estimasi dalam Juta Rupiah (OTR Jakarta)")

# Tampilkan dataset contoh
with st.expander("📊 Lihat Dataset Mobil Listrik 2026"):
    df = pd.read_csv("dataset.csv")
    st.dataframe(df, use_container_width=True)

st.markdown("---")
st.caption("Project SC 2026 - Prediksi Harga Mobil Listrik dengan ANN")
