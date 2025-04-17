from split_data import X_train, y_train, X_test
from sklearn.ensemble import RandomForestClassifier

# Buat model Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=123)

# Latih model
rf_model.fit(X_train, y_train)

# Prediksi pada data uji
y_pred = rf_model.predict(X_test)
