import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv(r"D:\ekstrak_fitur_1\ekstrak_fitur_1\dataset\phishing_1.csv")

# Memisahkan fitur dan label
X = df.drop(columns=['class'])  # Fitur (tanpa target)
y = df['class']  # Target (label phishing atau bukan)

# Membagi dataset: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

print(f"Data latih: {X_train.shape}, Data uji: {X_test.shape}")
