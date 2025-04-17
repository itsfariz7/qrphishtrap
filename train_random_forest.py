import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset phishing.csv
df = pd.read_csv (r"D:\ekstrak_fitur_1\ekstrak_fitur_1\dataset\phishing_1.csv")

# Pastikan semua fitur yang diekstrak ada dalam dataset
X = df.drop(columns=["class"])  # Fitur
y = df["class"]  # Target (Phishing atau Aman)

# Split data untuk pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Inisialisasi model Random Forest
model = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=123)
#estimator: membuat cabang pohon RF, max_dept: kedalaman pohon (None=tidak terbatas kedalamannya dalam membuat keputusan), random_state bisa berapa aja 

# Latih model
model.fit(X_train, y_train)

# Evaluasi model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print(f"Jumlah fitur saat training: {X_train.shape[1]}")

# Simpan model
joblib.dump(model, "random_forest_model.3.pkl")
print("Model berhasil disimpan sebagai random_forest_model.3.pkl")