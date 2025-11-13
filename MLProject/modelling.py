import os
import sys
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

    # === Ambil parameter dari CLI ===
    n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    dataset = sys.argv[3] if len(sys.argv) > 3 else "StudentsPerformance_preprocessing.joblib"

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, dataset)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset tidak ditemukan: {data_path}")

    # === Load dataset ===
    data = load(data_path)
    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]

    # === Setup MLflow tracking ===
    tracking_dir = os.path.join(base_dir, "mlruns")
    os.makedirs(tracking_dir, exist_ok=True)
    mlflow.set_tracking_uri(f"file://{tracking_dir}")

    experiment_name = "Student Performance Classification"
    mlflow.set_experiment(experiment_name)

    # === Mulai run MLflow ===
    with mlflow.start_run(run_name=f"RF_{n_estimators}_{max_depth}") as run:
        start_time = time.time()

        # === Training model ===
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        model.fit(X_train, y_train)
        training_time = time.time() - start_time

        # === Evaluasi model ===
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        try:
            logloss = log_loss(y_test, y_proba)
        except Exception:
            logloss = float("nan")
        bal_acc = balanced_accuracy_score(y_test, y_pred)

        # === Logging parameter dan metrik ===
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", 42)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("training_time", training_time)
        mlflow.log_metric("log_loss", logloss)
        mlflow.log_metric("balanced_accuracy", bal_acc)

        # === Folder artifact ===
        artifact_dir = os.path.join(base_dir, "artifacts", "model")
        os.makedirs(artifact_dir, exist_ok=True)

        # === Confusion Matrix ===
        disp = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
        plt.title("Confusion Matrix")
        cmatrix_path = os.path.join(artifact_dir, "confusion_matrix.png")
        plt.savefig(cmatrix_path)
        mlflow.log_artifact(cmatrix_path, artifact_path="model")
        plt.close()

        # === Log Loss ===
        plt.bar(["Log Loss"], [logloss if not np.isnan(logloss) else 0])
        plt.title("Log Loss")
        logloss_png = os.path.join(artifact_dir, "log_loss_plot.png")
        plt.savefig(logloss_png)
        mlflow.log_artifact(logloss_png, artifact_path="model")
        plt.close()

        # === Balanced Accuracy ===
        plt.bar(["Balanced Acc"], [bal_acc])
        plt.title("Balanced Accuracy")
        balacc_png = os.path.join(artifact_dir, "balanced_accuracy_plot.png")
        plt.savefig(balacc_png)
        mlflow.log_artifact(balacc_png, artifact_path="model")
        plt.close()

        # === ROC Curve (binary) ===
        try:
            if y_proba.shape[1] == 2:
                fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1], pos_label=model.classes_[1])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title("ROC Curve")
                plt.legend(loc="lower right")
                roc_png = os.path.join(artifact_dir, "roc_curve.png")
                plt.savefig(roc_png)
                mlflow.log_artifact(roc_png, artifact_path="model")
                plt.close()
        except Exception as e:
            print("ROC curve skipped:", e)

        # === Precision-Recall Curve (binary) ===
        try:
            if y_proba.shape[1] == 2:
                from sklearn.metrics import precision_recall_curve
                precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_proba[:, 1], pos_label=model.classes_[1])
                plt.plot(recall_curve, precision_curve)
                plt.xlabel("Recall")
                plt.ylabel("Precision")
                plt.title("Precision-Recall Curve")
                pr_png = os.path.join(artifact_dir, "precision_recall_curve.png")
                plt.savefig(pr_png)
                mlflow.log_artifact(pr_png, artifact_path="model")
                plt.close()
        except Exception as e:
            print("PR curve skipped:", e)

        # === Simpan metrik ke JSON ===
        metric_info = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "training_time": training_time,
            "log_loss": logloss,
            "balanced_accuracy": bal_acc
        }
        metric_path = os.path.join(artifact_dir, "metric_info.json")
        with open(metric_path, "w") as f:
            json.dump(metric_info, f, indent=4)
        mlflow.log_artifact(metric_path, artifact_path="model")

        # === HTML estimator ===
        estimator_html = os.path.join(artifact_dir, "estimator.html")
        with open(estimator_html, "w") as f:
            f.write(f"<h2>RandomForestClassifier</h2><p>n_estimators={n_estimators}, max_depth={max_depth}</p>")
        mlflow.log_artifact(estimator_html, artifact_path="model")

        # === Simpan model ===
        mlflow.sklearn.log_model(model, artifact_path="model")

        print(f"✅ Training selesai — n_estimators={n_estimators}, max_depth={max_depth}")
        print(f"   Akurasi: {accuracy:.4f}")
