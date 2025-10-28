import os
import time
import json
import warnings
import numpy as np
import matplotlib.pyplot as plt
from joblib import load
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay, log_loss, balanced_accuracy_score
import mlflow
import mlflow.sklearn
import sys

# 
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(42)

    # 
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, "StudentsPerformance_preprocessing.joblib")
    split_data = load(dataset_path)

    X_train = split_data["X_train"]
    X_test = split_data["X_test"]
    y_train = split_data["y_train"]
    y_test = split_data["y_test"]

    # 
    n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    # 
    mlflow.sklearn.autolog(log_input_examples=False, log_models=False)

    # 
    with mlflow.start_run():
        start_time = time.time()

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        model.fit(X_train, y_train)
        training_time = time.time() - start_time

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        logloss_val = log_loss(y_test, y_prob)
        bal_acc = balanced_accuracy_score(y_test, y_pred)

        # 
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("training_time", training_time)
        mlflow.log_metric("log_loss", logloss_val)
        mlflow.log_metric("balanced_accuracy", bal_acc)

        # 
        cmatrix = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
        plt.title("Confusion Matrix")
        cm_path = os.path.join(base_dir, "confusion_matrix.png")
        plt.savefig(cm_path)
        mlflow.log_artifact(cm_path)
        plt.close()

        # 
        metric_info = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "training_time": training_time,
            "log_loss": logloss_val,
            "balanced_accuracy": bal_acc
        }
        metric_path = os.path.join(base_dir, "metric_info.json")
        with open(metric_path, "w") as f:
            json.dump(metric_info, f, indent=4)
        mlflow.log_artifact(metric_path)

        #
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model"
        )

    print(f"\n✅ Model selesai dilatih dengan hyperparameter:")
    print(f"n_estimators = {n_estimators}")
    print(f"max_depth = {max_depth}")
    print(f"Akurasi: {acc:.4f}")