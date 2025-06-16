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

# Inisialisasi MLflow
mlflow.set_tracking_uri("file:///Users/promac/Documents/01_AI_MATERI/01_PROJEK/Eksperimen_SML_MohammadPrastya/mlruns")
mlflow.set_experiment("Eksperimen_SML_Mohammad_Nurdin_Prastya_Hermansah")

# Load data
data_filter = pd.read_csv("Membangun_model/netflix_preprocessing.csv")

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
    # Inisialisasi model RandomForest dengan parameter default
    rf = RandomForestClassifier(random_state=42)

    # Latih model
    rf.fit(X_train, y_train)

    # Prediksi
    y_pred = rf.predict(X_test)

    # Evaluasi komplit
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # Logging MLflow
    mlflow.log_param("model_type", "RandomForest_default")
    mlflow.log_metric("test_accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.sklearn.log_model(rf, "rf_cluster_model_default", input_example=X_train[:5])

    print("Model Default RandomForestClassifier digunakan.")
    print("Test Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("\nConfusion Matrix:\n", conf_matrix)
    print("\nClassification Report:\n", report)