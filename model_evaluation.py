from split_data import y_test  # Impor y_test dari split_data.py
from random_forest_model import y_pred  # Impor y_pred dari random_forest_model.py
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Hitung Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Lihat laporan akurasi
print("Confusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nAkurasi Model:", accuracy_score(y_test, y_pred))
