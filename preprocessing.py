import pandas as pd

# Load dataset
df = pd.read_csv(r"D:\ekstrak_fitur_1\ekstrak_fitur_1\dataset\phishing_1.csv")

# Pastikan semua fitur kecuali 'class' adalah numerik
X = df.drop(columns=['class'])  # Fitur (tanpa target)
y = df['class']  # Target (label phishing atau bukan)

# Tampilkan info data setelah ekstraksi
print(X.info())
print(y.value_counts())  # Melihat distribusi label
