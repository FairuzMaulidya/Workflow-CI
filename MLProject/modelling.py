# Mengimport Library
import os
import time
import json
import warnings
import numpy as np
import matplotlib.pyplot as plt
from joblib import load
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,ConfusionMatrixDisplay, log_loss, balanced_accuracy_score,roc_curve, auc, precision_recall_curve
import mlflow
import mlflow.sklearn
import sys

# Membuat main program
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(42)

    # Load dataset hasil preprocessing
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, "StudentsPerformance_preprocessing.joblib")
    split_data = load(dataset_path)

    X_train = split_data["X_train"]
    X_test = split_data["X_test"]
    y_train = split_data["y_train"]
    y_test = split_data["y_test"]

    # Inisialisasi hyperparameter
    n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    # Buat folder model
    model_dir = os.path.join(base_dir, "model")
    os.makedirs(model_dir, exist_ok=True)

    # Mulai MLflow run manual logging
    with mlflow.start_run():
        start_time = time.time()

        # Training model
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        model.fit(X_train, y_train)
        training_time = time.time() - start_time

        # Evaluasi model
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        logloss_val = log_loss(y_test, y_proba)
        bal_acc = balanced_accuracy_score(y_test, y_pred)

        # Logging parameter & metric
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", 42)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("training_time", training_time)
        mlflow.log_metric("log_loss", logloss_val)
        mlflow.log_metric("balanced_accuracy", bal_acc)

        # Confusion Matrix
        cmatrix = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
        plt.title("Confusion Matrix")
        cm_path = os.path.join(model_dir, "confusion_matrix.png")
        plt.savefig(cm_path)
        mlflow.log_artifact(cm_path)
        plt.close()

        # Log Loss plot
        plt.bar(["Log Loss"], [logloss_val])
        plt.title("Log Loss")
        logloss_png = os.path.join(model_dir, "log_loss_plot.png")
        plt.savefig(logloss_png)
        mlflow.log_artifact(logloss_png)
        plt.close()

        # Balanced Accuracy plot
        plt.bar(["Balanced Acc"], [bal_acc])
        plt.title("Balanced Accuracy")
        balacc_png = os.path.join(model_dir, "balanced_accuracy_plot.png")
        plt.savefig(balacc_png)
        mlflow.log_artifact(balacc_png)
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

        # Precision-Recall Curve
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_proba[:, 1], pos_label=model.classes_[1])
        plt.plot(recall_curve, precision_curve)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Training Precision-Recall Curve")
        pr_png = os.path.join(model_dir, "training_precision_recall_curve.png")
        plt.savefig(pr_png)
        mlflow.log_artifact(pr_png)
        plt.close()
        
        # Simpan metric_info.json
        metric_info = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "training_time": training_time,
            "log_loss": logloss_val,
            "balanced_accuracy": bal_acc
        }
        metric_path = os.path.join(model_dir, "metric_info.json")
        with open(metric_path, "w") as f:
            json.dump(metric_info, f, indent=4)
        mlflow.log_artifact(metric_path)

        #Membuat file estimator.html
        estimator_html = os.path.join(model_dir, "estimator.html")
        with open(estimator_html, "w") as f:
            f.write(f"<h2>RandomForestClassifier</h2><p>n_estimators={n_estimators}, max_depth={max_depth}, random_state=42</p>")
        mlflow.log_artifact(estimator_html)

        # Simpan model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model"
        )

    # Print hasil
    print("\nModel Terbaik")
    print(f"n_estimators = {n_estimators}")
    print(f"max_depth = {max_depth}")
    print(f"Akurasi: {accuracy:.4f}")