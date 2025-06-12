import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Tentukan experiment dan tracking URI
mlflow.set_experiment("rekrutmen_experiment")
base_dir = os.path.dirname(os.path.abspath(__file__))
mlruns_path = os.path.join(base_dir, "mlruns")
mlflow.set_tracking_uri(f"file://{mlruns_path}")

# Autolog untuk MLflow
mlflow.sklearn.autolog()

# Baca dataset
csv_path = os.path.join(base_dir, "recruitment_data_clean.csv")
df = pd.read_csv(csv_path)

# Label encoding untuk fitur kategorikal
label_cols = df.select_dtypes(include='object').columns
for col in label_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

# Pisahkan fitur dan target
X = df.drop(columns=["HiringDecision"])
y = df["HiringDecision"]

print("Fitur:", X.columns.tolist())
print("Jumlah fitur:", X.shape[1])


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Prediksi
y_pred = model.predict(X_test)

# Evaluasi
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Cetak hasil
print("✅ Model dilatih dan dicatat di MLflow.")
print(f"🔍 Akurasi: {accuracy:.4f}")
print(f"🔍 Presisi: {precision:.4f}")
print(f"🔍 Recall: {recall:.4f}")
print(f"🔍 F1-Score: {f1:.4f}")
