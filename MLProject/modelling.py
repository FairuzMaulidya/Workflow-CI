import os
import time
import json
import warnings
import numpy as np
import matplotlib.pyplot as plt
from joblib import load
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    ConfusionMatrixDisplay, log_loss, balanced_accuracy_score,
    roc_curve, auc, precision_recall_curve
)
import mlflow
import mlflow.sklearn

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(42)

    # Pastikan path relatif agar bisa di CI dan lokal
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "StudentsPerformance_preprocessing.joblib")

    # Pastikan file dataset ada
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset tidak ditemukan: {data_path}")

    data = load(data_path)
    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]

    # Hyperparameter Grid
    n_est = np.linspace(50, 300, 3, dtype=int)
    max_dep = np.linspace(5, 20, 3, dtype=int)

    best_accuracy = 0
    best_params = {}

    # FIX: gunakan path relatif (CI-friendly)
    tracking_dir = os.path.join(base_dir, "mlruns")
    os.makedirs(tracking_dir, exist_ok=True)
    mlflow.set_tracking_uri(f"file://{tracking_dir}")
    mlflow.set_experiment("Student Performance Classification")

    # Jalankan main run agar terstruktur di MLflow
    with mlflow.start_run(run_name="Main_Run"):
        for n_estimators in n_est:
            for max_depth in max_dep:
                with mlflow.start_run(nested=True):
                    start_time = time.time()

                    model = RandomForestClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        random_state=42
                    )
                    model.fit(X_train, y_train)
                    training_time = time.time() - start_time

                    y_pred = model.predict(X_test)
                    y_proba = model.predict_proba(X_test)

                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
                    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
                    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
                    logloss = log_loss(y_test, y_proba)
                    bal_acc = balanced_accuracy_score(y_test, y_pred)

                    # Log parameter & metrics
                    mlflow.log_params({
                        "n_estimators": n_estimators,
                        "max_depth": max_depth,
                        "random_state": 42
                    })
                    mlflow.log_metrics({
                        "accuracy": accuracy,
                        "precision": precision,
                        "recall": recall,
                        "f1_score": f1,
                        "training_time": training_time,
                        "log_loss": logloss,
                        "balanced_accuracy": bal_acc
                    })

                    # Folder model output
                    model_dir = os.path.join(base_dir, "model")
                    os.makedirs(model_dir, exist_ok=True)

                    # Confusion Matrix
                    cmatrix = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
                    plt.title("Training Confusion Matrix")
                    cmatrix_path = os.path.join(model_dir, "training_confusion_matrix.png")
                    plt.savefig(cmatrix_path)
                    mlflow.log_artifact(cmatrix_path)
                    plt.close()

                    # ROC Curve
                    fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1], pos_label=model.classes_[1])
                    roc_auc = auc(fpr, tpr)
                    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
                    plt.xlabel("False Positive Rate")
                    plt.ylabel("True Positive Rate")
                    plt.title("Training ROC Curve")
                    plt.legend(loc="lower right")
                    roc_png = os.path.join(model_dir, "training_roc_curve.png")
                    plt.savefig(roc_png)
                    mlflow.log_artifact(roc_png)
                    plt.close()

                    # Simpan metrik dan model
                    metric_info = {
                        "accuracy": accuracy,
                        "precision": precision,
                        "recall": recall,
                        "f1_score": f1,
                        "training_time": training_time,
                        "log_loss": logloss,
                        "balanced_accuracy": bal_acc
                    }
                    with open(os.path.join(model_dir, "metric_info.json"), "w") as f:
                        json.dump(metric_info, f, indent=4)

                    mlflow.log_artifact(os.path.join(model_dir, "metric_info.json"))
                    mlflow.sklearn.log_model(model, artifact_path="model")

                    print(f" Run selesai: n_estimators={n_estimators}, max_depth={max_depth}, acc={accuracy:.4f}")

                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_params = {"n_estimators": n_estimators, "max_depth": max_depth}

    print(f"\nModel terbaik: {best_params}")
    print(f"Akurasi terbaik: {best_accuracy:.4f}")
