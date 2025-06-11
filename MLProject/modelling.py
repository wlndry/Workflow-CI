import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns

# MLflow autolog
mlflow.sklearn.autolog()

# Load data
df = pd.read_csv("recruitment_data_clean.csv")

# Cek nama kolom
print("Kolom:", df.columns.tolist())

# Pisahkan fitur dan target
X = df.drop(columns='HiringDecision')  # Ganti sesuai nama target
y = df['HiringDecision']

# Split data dengan stratifikasi
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# MLflow run dengan nama
with mlflow.start_run(run_name="RandomForest_Rekrutmen_1"):

    # Inisialisasi dan training model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Prediksi
    y_pred = model.predict(X_test)

    # Hitung metrik
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # Print metrik
    print(f"Akurasi: {acc:.4f}")
    print(f"Presisi: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-score: {f1:.4f}")

    # Logging metrik secara manual
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)

    # Confusion matrix plot
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    # Simpan gambar
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
