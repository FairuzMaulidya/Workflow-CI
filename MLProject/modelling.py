#!/usr/bin/env python3
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
from sklearn.preprocessing import label_binarize
import mlflow
import mlflow.sklearn

def safe_mkdir(path):
    os.makedirs(path, exist_ok=True)

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(42)

    # --- Arguments ---
    n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    dataset = sys.argv[3] if len(sys.argv) > 3 else "StudentsPerformance_preprocessing.joblib"

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, dataset)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    # === Load dataset ===
    data = load(data_path)
    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]

    # === Setup MLflow tracking ===
    # Prefer env var MLFLOW_TRACKING_URI (CI may set it). If not set, use local mlruns folder.
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
    if mlflow_uri:
        # if env var like "file:mlruns" -> convert to file:// absolute location if relative
        if mlflow_uri.startswith("file:") and not mlflow_uri.startswith("file://"):
            # e.g. file:mlruns -> file://<abs_path>/mlruns
            rel = mlflow_uri.split("file:", 1)[1]
            rel = rel if rel else "mlruns"
            abs_path = os.path.abspath(os.path.join(base_dir, rel))
            mlflow.set_tracking_uri(f"file://{abs_path}")
        else:
            mlflow.set_tracking_uri(mlflow_uri)
    else:
        tracking_dir = os.path.join(base_dir, "mlruns")
        safe_mkdir(tracking_dir)
        mlflow.set_tracking_uri(f"file://{tracking_dir}")

    experiment_name = "Student Performance Classification"
    mlflow.set_experiment(experiment_name)

    # Use a run context (start a run; nested runs are allowed when invoked by mlflow run)
    with mlflow.start_run(run_name=f"RF_{n_estimators}_{max_depth}") as run:
        start_time = time.time()

        # === Training ===
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        model.fit(X_train, y_train)
        training_time = time.time() - start_time

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        # === Metrics ===
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        try:
            logloss = log_loss(y_test, y_proba)
        except Exception:
            logloss = float("nan")
        bal_acc = balanced_accuracy_score(y_test, y_pred)

        # Log params & metrics
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", 42)

        mlflow.log_metric("accuracy", float(accuracy))
        mlflow.log_metric("precision", float(precision))
        mlflow.log_metric("recall", float(recall))
        mlflow.log_metric("f1_score", float(f1))
        mlflow.log_metric("training_time", float(training_time))
        mlflow.log_metric("log_loss", float(logloss) if not np.isnan(logloss) else -1.0)
        mlflow.log_metric("balanced_accuracy", float(bal_acc))

        # === Artifacts: save plots into artifacts/model and log them ===
        artifact_root = os.path.join(base_dir, "artifacts")
        model_artifact_dir = os.path.join(artifact_root, "model")  # IMPORTANT: artifact path must be "model"
        safe_mkdir(model_artifact_dir)

        # Confusion matrix
        disp = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
        plt.title("Confusion Matrix")
        cm_path = os.path.join(model_artifact_dir, f"confusion_matrix_{n_estimators}_{max_depth}.png")
        plt.savefig(cm_path)
        plt.close()
        mlflow.log_artifact(cm_path, artifact_path="model")

        # Log Loss bar
        plt.figure()
        plt.bar(["Log Loss"], [logloss if not np.isnan(logloss) else 0])
        plt.title("Log Loss")
        logloss_png = os.path.join(model_artifact_dir, "log_loss_plot.png")
        plt.savefig(logloss_png)
        plt.close()
        mlflow.log_artifact(logloss_png, artifact_path="model")

        # Balanced Accuracy bar
        plt.figure()
        plt.bar(["Balanced Acc"], [bal_acc])
        plt.title("Balanced Accuracy")
        balacc_png = os.path.join(model_artifact_dir, "balanced_accuracy_plot.png")
        plt.savefig(balacc_png)
        plt.close()
        mlflow.log_artifact(balacc_png, artifact_path="model")

        # ROC & Precision-Recall: handle binary and multiclass safely
        try:
            n_classes = y_proba.shape[1]
            if n_classes == 2:
                # binary
                fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1], pos_label=model.classes_[1])
                roc_auc = auc(fpr, tpr)
                plt.figure()
                plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title("ROC Curve")
                plt.legend(loc="lower right")
                roc_png = os.path.join(model_artifact_dir, "roc_curve.png")
                plt.savefig(roc_png)
                plt.close()
                mlflow.log_artifact(roc_png, artifact_path="model")

                # PR curve
                precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_proba[:, 1], pos_label=model.classes_[1])
                plt.figure()
                plt.plot(recall_curve, precision_curve)
                plt.xlabel("Recall")
                plt.ylabel("Precision")
                plt.title("Precision-Recall Curve")
                pr_png = os.path.join(model_artifact_dir, "precision_recall_curve.png")
                plt.savefig(pr_png)
                plt.close()
                mlflow.log_artifact(pr_png, artifact_path="model")
            else:
                # multiclass: compute micro-average ROC
                y_test_bin = label_binarize(y_test, classes=model.classes_)
                # handle shapes
                if y_test_bin.shape[1] == y_proba.shape[1]:
                    fpr = dict()
                    tpr = dict()
                    roc_auc = dict()
                    for i in range(y_proba.shape[1]):
                        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
                        roc_auc[i] = auc(fpr[i], tpr[i])
                    # micro
                    fpr_micro, tpr_micro, _ = roc_curve(y_test_bin.ravel(), y_proba.ravel())
                    roc_auc_micro = auc(fpr_micro, tpr_micro)
                    plt.figure()
                    plt.plot(fpr_micro, tpr_micro, label=f"ROC curve (micro) AUC = {roc_auc_micro:.2f}")
                    plt.xlabel("False Positive Rate")
                    plt.ylabel("True Positive Rate")
                    plt.title("ROC Curve (micro-average)")
                    plt.legend(loc="lower right")
                    roc_png = os.path.join(model_artifact_dir, "roc_curve_micro.png")
                    plt.savefig(roc_png)
                    plt.close()
                    mlflow.log_artifact(roc_png, artifact_path="model")
                # skip PR for multiclass to keep simple
        except Exception as e:
            print("Warning: ROC/PR plotting skipped due to:", e)

        # metric_info.json
        metric_info = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "training_time": float(training_time),
            "log_loss": float(logloss) if not np.isnan(logloss) else None,
            "balanced_accuracy": float(bal_acc)
        }
        metric_path = os.path.join(model_artifact_dir, "metric_info.json")
        with open(metric_path, "w") as f:
            json.dump(metric_info, f, indent=4)
        mlflow.log_artifact(metric_path, artifact_path="model")

        # estimator.html
        estimator_html = os.path.join(model_artifact_dir, "estimator.html")
        with open(estimator_html, "w") as f:
            f.write(f"<h2>RandomForestClassifier</h2><p>n_estimators={n_estimators}, max_depth={max_depth}, random_state=42</p>")
        mlflow.log_artifact(estimator_html, artifact_path="model")

        # === LOG MODEL ===
        # IMPORTANT: artifact_path must be "model" so downstream steps can find artifacts/model/MLmodel
        mlflow.sklearn.log_model(sk_model=model, artifact_path="model")

        print(f"✅ Training selesai — n_estimators={n_estimators}, max_depth={max_depth}")
        print(f"   Akurasi: {accuracy:.4f}")
