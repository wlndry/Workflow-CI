import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import mlflow
import mlflow.sklearn
import dagshub
import seaborn as sns
import matplotlib.pyplot as plt
import os
import tempfile

# Inisialisasi DagsHub
dagshub.init(repo_owner="wlndry", repo_name="smsml-project", mlflow=True)

# Set tracking URI ke DagsHub
mlflow.set_tracking_uri("https://dagshub.com/wlndry/smsml-project.mlflow")


# Load data
df = pd.read_csv("recruitment_data_clean.csv")
X = df.drop(columns='HiringDecision')
y = df['HiringDecision']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Simpan akurasi terbaik
best_accuracy = 0.0

# Daftar kombinasi hyperparameter (contoh tuning sederhana)
param_grid = [
    {"n_estimators": 100, "max_depth": 5, "random_state": 42},
    {"n_estimators": 150, "max_depth": 7, "random_state": 42},
    {"n_estimators": 200, "max_depth": 10, "random_state": 42},
]

# Loop training
for params in param_grid:
    with mlflow.start_run(run_name="RF_ManualLogging"):
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)

        # Logging manual
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", report['1']['precision'])
        mlflow.log_metric("recall", report['1']['recall'])
        mlflow.log_metric("f1_score", report['1']['f1-score'])

        # Simpan artefak tambahan
        with tempfile.TemporaryDirectory() as tmpdir:
            # Confusion Matrix
            plt.figure(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            cm_path = os.path.join(tmpdir, "confusion_matrix.png")
            plt.savefig(cm_path)
            mlflow.log_artifact(cm_path)

            # Classification Report
            report_df = pd.DataFrame(report).transpose()
            report_path = os.path.join(tmpdir, "classification_report.csv")
            report_df.to_csv(report_path)
            mlflow.log_artifact(report_path)

        # Simpan model hanya jika lebih baik dari best_accuracy
        if acc > best_accuracy:
            best_accuracy = acc
            print(f"Model baru dengan akurasi lebih tinggi ditemukan: {acc}")
            mlflow.sklearn.log_model(model, artifact_path="best_model")
        else:
            print(f"Model dengan akurasi {acc} tidak lebih baik dari sebelumnya ({best_accuracy})")
