from phising_detector import extract_features
import joblib
import pandas as pd

# Load model yang sudah disimpan
model_path = r"D:\ekstrak_fitur_1\ekstrak_fitur_1\random_forest_model.3.pkl"
model = joblib.load(model_path)

# URL yang akan diuji
url_test = "http://exemple.gistemp.com/mail/en.php"

# Ekstraksi fitur dari URL
features = extract_features(url_test)

# Konversi fitur ke dalam format DataFrame
X_input = pd.DataFrame([features])

# Ambil nama fitur yang digunakan saat pelatihan
expected_features = model.feature_names_in_

# **Normalisasi nama fitur agar sesuai dengan model**
rename_mapping = {
    "PrefixSuffix": "PrefixSuffix-",  # Sesuaikan nama agar cocok
    # Tambahkan pemetaan lain jika ada
}

X_input.rename(columns=rename_mapping, inplace=True)

# **Tambahkan fitur 'Index' jika tidak ada (isi dengan 0)**
if "Index" not in X_input.columns:
    X_input["Index"] = 0  # Atau isi dengan nilai yang sesuai jika ada di dataset asli


# **Pastikan input hanya memiliki fitur yang sesuai dengan model**
X_input = X_input.reindex(columns=expected_features, fill_value=0)

# **Prediksi menggunakan model**
prediction = model.predict(X_input)

# **Interpretasi hasil prediksi**
if prediction[0] == -1:
    print("URL ini adalah phishing")
else:
    print("URL ini aman")
