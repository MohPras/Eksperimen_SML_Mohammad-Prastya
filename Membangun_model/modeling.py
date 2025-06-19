import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Inisialisasi MLflow ke localhost (Tracking UI)
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Eksperimen_SML_Mohammad_Nurdin_Prastya_Hermansah")
mlflow.autolog()  # ✅ Autolog aktif, tanpa manual log

# Load data
data_filter = pd.read_csv("preprocesing/netflix_preprocessing.csv")

# Pisahkan fitur dan target
X = data_filter.drop(columns=['cluster'])
y = data_filter['cluster']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set shape: X_train={X_train.shape}, y_train={y_train.shape}")
print(f"Test set shape: X_test={X_test.shape}, y_test={y_test.shape}")

# Jalankan MLflow
with mlflow.start_run(run_name="Modeling_Tanpa_Tuning"):
    # Inisialisasi dan latih model
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)

    # Prediksi & evaluasi
    y_pred = rf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # Tampilkan hasil (logging otomatis oleh autolog)
    print("Model Default RandomForestClassifier digunakan.")
    print("Test Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("\nConfusion Matrix:\n", conf_matrix)
    print("\nClassification Report:\n", report)