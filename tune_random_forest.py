import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset phishing.csv
df = pd.read_csv(r"D:\ekstrak_fitur_1\ekstrak_fitur_1\dataset\phishing_1.csv")

# Pisahkan fitur dan label
X = df.drop(columns=["class"])
y = df["class"]

# Split data untuk training & testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Definisikan rentang jumlah estimator
param_grid = {'n_estimators': [100, 200, 300, 500]}

# Grid Search untuk mencari nilai terbaik
grid_search = GridSearchCV(RandomForestClassifier(random_state=42),
                           param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# Latih model dengan Grid Search
grid_search.fit(X_train, y_train)

# Cetak nilai terbaik
print(f"Best n_estimators: {grid_search.best_params_['n_estimators']}")
