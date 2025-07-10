# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model, scaler, dan kolom fitur
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
training_columns = joblib.load("training_columns.pkl")

st.set_page_config(page_title="Prediksi Resign", layout="centered")
st.title("Prediksi Karyawan Resign (Attrition Prediction)")
st.markdown("Masukkan data karyawan di bawah ini:")

# Input fitur penting
Age = st.slider("Umur", 18, 60, 30)
MonthlyIncome = st.number_input("Gaji Bulanan", 1000, 20000, 5000)
TotalWorkingYears = st.slider("Total Tahun Bekerja", 0, 40, 5)
YearsAtCompany = st.slider("Tahun di Perusahaan", 0, 40, 3)
JobSatisfaction = st.selectbox("Kepuasan Kerja", [1, 2, 3, 4])
EnvironmentSatisfaction = st.selectbox("Kepuasan Lingkungan", [1, 2, 3, 4])
WorkLifeBalance = st.selectbox("Work-Life Balance", [1, 2, 3, 4])
DistanceFromHome = st.slider("Jarak dari Rumah (mil)", 1, 30, 10)

# Buat dataframe kosong lalu isi nilai input
input_data = pd.DataFrame(np.zeros((1, len(training_columns))), columns=training_columns)
input_data.at[0, 'Age'] = Age
input_data.at[0, 'MonthlyIncome'] = MonthlyIncome
input_data.at[0, 'TotalWorkingYears'] = TotalWorkingYears
input_data.at[0, 'YearsAtCompany'] = YearsAtCompany
input_data.at[0, 'JobSatisfaction'] = JobSatisfaction
input_data.at[0, 'EnvironmentSatisfaction'] = EnvironmentSatisfaction
input_data.at[0, 'WorkLifeBalance'] = WorkLifeBalance
input_data.at[0, 'DistanceFromHome'] = DistanceFromHome

input_data = input_data.astype(float)
input_scaled = scaler.transform(input_data)
prediction = model.predict(input_scaled)

if st.button("Prediksi"):
    if prediction[0] == 1:
        st.error("🔴 Karyawan ini berpotensi **resign** 😥")
    else:
        st.success("🟢 Karyawan ini kemungkinan **bertahan** 😊")
