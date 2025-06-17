import argparse
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

def main(csv_url):
    # Load dan persiapkan data
    data_filter = pd.read_csv(csv_url)

    # Pisahkan fitur dan target
    X = data_filter.drop(columns=['cluster'])
    y = data_filter['cluster']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Training set shape: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"Test set shape: X_test={X_test.shape}, y_test={y_test.shape}")

    # Parameter grid
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2']
    }

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

        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_cv_score = grid_search.best_score_

        y_pred = best_model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        conf_matrix = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        mlflow.log_params(best_params)
        mlflow.log_metric("best_cv_score", best_cv_score)
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(best_model, "model", input_example=X_train[:5])

        print("Model RandomForestClassifier dengan Tuning digunakan.")
        print("Best Parameters:", best_params)
        print("Best CV Score:", best_cv_score)
        print("Test Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)
        print("\nConfusion Matrix:\n", conf_matrix)
        print("\nClassification Report:\n", report)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_url", type=str, required=True)
    args = parser.parse_args()

    main(args.csv_url)