import streamlit as st
import pandas as pd
import requests

st.title("🎬 Prediksi Data Netflix")

st.markdown("Upload file `.csv` atau masukkan data manual di bawah untuk prediksi.")

option = st.radio("Pilih metode input:", ["Upload CSV", "Input Manual"])

df = None

if option == "Upload CSV":
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("📄 Data yang diupload:")
        st.dataframe(df)

elif option == "Input Manual":
    # Contoh input 2 kolom: PC1 dan PC2
    num_rows = st.number_input("Jumlah baris input:", min_value=1, value=1)
    data_input = []

    st.markdown("Masukkan nilai fitur:")

    for i in range(num_rows):
        with st.expander(f"Baris {i+1}"):
            pc1 = st.number_input(f"PC1 (baris {i+1})", key=f"pc1_{i}")
            pc2 = st.number_input(f"PC2 (baris {i+1})", key=f"pc2_{i}")
            data_input.append([pc1, pc2])

    df = pd.DataFrame(data_input, columns=["PC1", "PC2"])
    st.write("📋 Data yang dimasukkan:")
    st.dataframe(df)

# Tombol Prediksi
if df is not None and st.button("🚀 Prediksi"):
    input_data = {
        "dataframe_split": {
            "columns": df.columns.tolist(),
            "data": df.values.tolist()
        }
    }

    try:
        # ---- BAGIAN YANG DIPERBAIKI ----
        # Menggunakan port 5005 karena ini adalah port yang diekspos oleh Docker
        # dan dipetakan ke port 8000 di dalam container.
        response = requests.post("http://127.0.0.1:5005/predict", json=input_data)
        # -------------------------------

        if response.status_code == 200:
            st.success("✅ Prediksi berhasil!")
            st.write("📊 Hasil Prediksi:")
            st.json(response.json())
        else:
            # Menampilkan detail error dari respons server jika status_code bukan 200
            st.error(f"❌ Error {response.status_code}: {response.reason}")
            try:
                st.json(response.json()) # Coba parse sebagai JSON
            except ValueError:
                st.text(response.text) # Jika bukan JSON, tampilkan sebagai teks biasa
    except requests.exceptions.ConnectionError as e:
        st.error(f"❌ Terjadi kesalahan koneksi: Pastikan server Docker berjalan dan dapat diakses di http://127.0.0.1:5005. Detail: {e}")
    except Exception as e:
        st.error(f"❌ Terjadi kesalahan tak terduga: {e}")