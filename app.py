import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Prediksi Harga Mobil", page_icon="🚗", layout="centered"
)

st.title("🚗 Aplikasi Prediksi Harga Mobil Bekas")
st.write("Masukkan spesifikasi mobil di bawah ini untuk memprediksi harganya:")


# Load Model, Scaler & Column Names
@st.cache_resource
def load_assets():
  model = joblib.load("model_mobil.pkl")
  scaler = joblib.load("scaler_mobil.pkl")
  feature_cols = joblib.load("feature_columns.pkl")
  return model, scaler, feature_cols


try:
  model, scaler, feature_cols = load_assets()
  st.success("Model berhasil dimuat!")
except Exception as e:
  st.error(
      "Gagal memuat file .pkl. Pastikan Anda sudah menjalankan 'python"
      f" train_lokal.py'! Error: {e}"
  )

st.subheader("📋 Masukkan Spesifikasi Mobil")
col1, col2 = st.columns(2)

with col1:
  horsepower = st.number_input("Horsepower (HP)", 40, 300, 100)
  engine_size = st.number_input("Engine Size (cc)", 50, 400, 130)
  curb_weight = st.number_input("Berat Kendaraan (lbs)", 1000, 5000, 2500)

with col2:
  city_mpg = st.number_input("Konsumsi BBM Kota (MPG)", 5, 60, 25)
  highway_mpg = st.number_input("Konsumsi BBM Tol (MPG)", 5, 60, 30)

if st.button("💰 Hitung Prediksi Harga", type="primary"):
  try:
    # Buat dataframe input dengan struktur kolom yang persis sama dengan model
    input_df = pd.DataFrame(0.0, index=[0], columns=feature_cols)

    # Masukkan nilai input pengguna
    if "horsepower" in input_df.columns:
      input_df["horsepower"] = horsepower
    if "engine-size" in input_df.columns:
      input_df["engine-size"] = engine_size
    if "curb-weight" in input_df.columns:
      input_df["curb-weight"] = curb_weight
    if "city-mpg" in input_df.columns:
      input_df["city-mpg"] = city_mpg
    if "highway-mpg" in input_df.columns:
      input_df["highway-mpg"] = highway_mpg

    # Transformasi & Prediksi
    input_scaled = scaler.transform(input_df)
    prediksi = model.predict(input_scaled)[0]

    st.success(f"### Estimasi Harga Mobil: **${prediksi:,.2f}**")
  except Exception as e:
    st.error(f"Terjadi kesalahan saat memprediksi: {e}")