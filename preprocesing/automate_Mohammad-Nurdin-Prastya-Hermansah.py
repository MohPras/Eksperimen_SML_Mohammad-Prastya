import pandas as pd
import numpy as np
import mlflow
from joblib import load
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
import matplotlib.pyplot as plt
import os
import sys

mlflow.set_tracking_uri("file:///Users/promac/Documents/01_AI_MATERI/01_PROJEK/Eksperimen_SML_MohammadPrastya/mlruns")
mlflow.set_experiment("Eksperimen_SML_Mohammad_Nurdin_Prastya_Hermansah")

# ===============================
# DEFINISI CLASS OutlierHandler
# ===============================
class OutlierHandler(BaseEstimator, TransformerMixin):
    def __init__(self, method="zscore", threshold=3):
        self.method = method
        self.threshold = threshold

    def fit(self, X, y=None):
        if self.method == "zscore":
            self.mean_ = X.mean()
            self.std_ = X.std()
        return self

    def transform(self, X):
        if self.method == "zscore":
            z_scores = (X - self.mean_) / self.std_
            X_clean = np.where(np.abs(z_scores) > self.threshold, self.mean_, X)
            return pd.DataFrame(X_clean, columns=X.columns, index=X.index)
        return X

# ===============================
# Mulai MLflow Run Utama
# ===============================
with mlflow.start_run(run_name="Preprocessing Otomatis"):

    # ===============================
    # Load Dataset
    # ===============================
    df = pd.read_csv("netflix_raw/Genre_netflix.csv")
    mlflow.log_param("input_file", "Genre_netflix.csv")
    mlflow.log_metric("original_rows", len(df))
    mlflow.log_metric("original_columns", len(df.columns))

    # ===============================
    # Extract Duration
    # ===============================
    def extract_duration_numeric(df):
        df['duration_numeric'] = df['duration'].str.extract(r'(\d+)').astype(float)
        mlflow.log_param("duration_extraction", "regex_digits")
        mlflow.log_metric("duration_missing_values", df['duration_numeric'].isna().sum())
        df.drop(columns=['duration'], inplace=True)
        print("✅ Durasi berhasil diekstrak.")
        return df

    df = extract_duration_numeric(df)
    
    # ===============================
    # Delete Kolom
    # ===============================
    df.drop(['Unnamed: 0', 'title'], axis=1, inplace=True)

    # ===============================
    # Convert date_added to datetime
    # ===============================
    def process_date_column(df, column='date_added'):
        df[column] = df[column].astype(str).str.strip()
        df[column] = pd.to_datetime(df[column], format='%B %d, %Y', errors='coerce')
        df['year'] = df[column].dt.year
        df['month'] = df[column].dt.month
        df['day'] = df[column].dt.day
        df.drop(columns=[column], inplace=True)
        print("✅ Kolom date_added berhasil dikonversi ke datetime.")
        return df

    df = process_date_column(df)

    # ===============================
    # Load Encoder & Scaler (pkl)
    # ===============================
    def load_pkl(path):
        return load(path)

    type_encoder = load_pkl("preprocesing/prepocesing_pkl/type_label_encoder.pkl")
    rating_encoder = load_pkl("preprocesing/prepocesing_pkl/rating_encoder.pkl")
    cluster_encoder = load_pkl("preprocesing/prepocesing_pkl/cluster_label_encoder.pkl")
    country_encoder = load_pkl("preprocesing/prepocesing_pkl/country_binary_encoder.pkl")
    listedin_encoder = load_pkl("preprocesing/prepocesing_pkl/listedin_binary_encoder.pkl")
    scaler = load_pkl("preprocesing/prepocesing_pkl/minmax_scaler.pkl")

    mlflow.log_param("encoder_loaded", True)

    # ===============================
    # Apply Encoders
    # ===============================
    df['type'] = type_encoder.transform(df['type'])
    df['rating'] = rating_encoder.transform(df[['rating']]).ravel()
    df['cluster'] = cluster_encoder.transform(df['cluster'])

    country_encoded = country_encoder.transform(df[['country']])
    listedin_encoded = listedin_encoder.transform(df[['listed_in']])
    df.drop(columns=['country', 'listed_in'], inplace=True)
    df = pd.concat([df, country_encoded, listedin_encoded], axis=1)
    print("✅ Label encoding & binary encoding diterapkan.")

    # ===============================
    # Handle Outliers
    # ===============================
    df_numeric = df.select_dtypes(include='number')
    outlier_handler = OutlierHandler(method="zscore", threshold=3)
    outlier_handler.fit(df_numeric)
    df_cleaned = outlier_handler.transform(df_numeric)
    df[df_numeric.columns] = df_cleaned
    mlflow.log_param("outlier_handler_applied", True)
    mlflow.log_param("outlier_method", outlier_handler.method)
    mlflow.log_param("outlier_threshold", outlier_handler.threshold)
    print("✅ Outlier handling diterapkan tanpa file .pkl.")

    # ===============================
    # Scaling 
    # ===============================
    numeric_cols_for_scaling = scaler.feature_names_in_  # kolom saat fit scaler
    df_scaled = scaler.transform(df[numeric_cols_for_scaling])
    df[numeric_cols_for_scaling] = df_scaled
    mlflow.log_param("scaling_applied", True)
    print("✅ Scaling selesai.")

    # ===============================
    # PCA Transformasi
    # ===============================
    pca = load("preprocesing/prepocesing_pkl/pca_model.pkl")

    # Ambil hanya kolom yang cocok saat PCA dilatih
    pca_input_cols = pca.feature_names_in_
    df_for_pca = df[pca_input_cols]  # pastikan urutan & nama kolom sesuai

    df_pca = pca.transform(df_for_pca)

    # Buat kolom PCA baru
    pca_columns = [f'PC{i+1}' for i in range(df_pca.shape[1])]
    df_pca = pd.DataFrame(df_pca, columns=pca_columns, index=df.index)

    # Hapus kolom original (input ke PCA) dan ganti dengan hasil PCA
    df.drop(columns=pca_input_cols, inplace=True)
    df = pd.concat([df, df_pca], axis=1)

    mlflow.log_param("pca_applied", True)
    mlflow.log_param("pca_components", pca.n_components_)
    print("✅ PCA transformasi diterapkan.")


    # ===============================
    # Save Final Preprocessed Data
    # ===============================
    output_path = "Membangun_model/netflix_preprocessing.csv"  # arahkan ke dalam folder
    df.to_csv(output_path, index=False)

    mlflow.log_artifact(output_path)  # tetap bisa log file ini
    mlflow.log_param("output_file", output_path)

    print(f"✅ Preprocessing selesai. Data disimpan sebagai '{output_path}'")