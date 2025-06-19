import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

# Aktifkan autolog
mlflow.sklearn.autolog()

# Set ke tracking server di localhost
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Eksperimen_SML_Mohammad_Nurdin_Prastya_Hermansah")

# Load data hasil preprocessing
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

# Parameter grid untuk tuning
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2']
}

# Mulai experiment run (autolog akan otomatis mencatat semuanya)
with mlflow.start_run(run_name="Modeling_Dengan_Tuning"):
    rf = RandomForestClassifier(random_state=42)

    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    # Evaluasi manual untuk ditampilkan di terminal
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    print("✅ RandomForestClassifier dengan Tuning selesai dijalankan.")
    print("📌 Best Parameters:", grid_search.best_params_)
    print("📌 Classification Report:\n", classification_report(y_test, y_pred))
    print("📌 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))