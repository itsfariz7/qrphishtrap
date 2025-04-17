import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

# Load dataset
file_path = r"D:\ekstrak_fitur_1\ekstrak_fitur_1\dataset\phishing_1.csv"
df = pd.read_csv(file_path)

# Pastikan semua fitur yang diekstrak ada dalam dataset
X = df.drop(columns=["class"])  # Fitur
y = df["class"]  # Target (Phishing atau Aman)

# Split data untuk pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Definisikan rentang max_depth yang ingin diuji
param_grid = {'max_depth': [5, 10, 15, 20, None]}

# Grid Search untuk mencari nilai terbaik
grid_search = GridSearchCV(RandomForestClassifier(n_estimators=1000, random_state=42),
                           param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# Latih model dengan Grid Search
grid_search.fit(X_train, y_train)

# Cetak nilai max_depth terbaik
print(f"Best max_depth: {grid_search.best_params_['max_depth']}")

# Evaluasi model terbaik
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Optimized Model Accuracy: {accuracy:.2f}")

# Simpan model terbaik
joblib.dump(best_model, "optimized_random_forest.pkl")
print("Optimized model berhasil disimpan sebagai optimized_random_forest.pkl")
