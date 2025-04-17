df = pd.read_csv (r"D:\ekstrak_fitur_1\ekstrak_fitur_1\dataset\phishing_1.csv")
print(df.info())

from IPython.display import display
display(pd.DataFrame(df.columns, columns=["Nama Kolom"]))

# Pastikan semua fitur kecuali 'class' adalah numerik
X = df.drop(columns=['class'])  # Fitur (tanpa target)
y = df['class']  # Target (label phishing atau bukan)

# Tampilkan info data setelah ekstraksi
print(X.info())
print(y.value_counts())  # Melihat distribusi label

from sklearn.model_selection import train_test_split

# Membagi dataset: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

print(f"Data latih: {X_train.shape}, Data uji: {X_test.shape}")

from sklearn.ensemble import RandomForestClassifier

# Buat model Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=123)

# Latih model
rf_model.fit(X_train, y_train)

# Prediksi pada data uji
y_pred = rf_model.predict(X_test)

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Hitung Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Lihat laporan akurasi
print("Confusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nAkurasi Model:", accuracy_score(y_test, y_pred))

import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset phishing.csv
df = pd.read_csv (r"C:\Users\Lenovo\Documents\TUGAS TUGAS\dataPhishing\phishing_1.csv")

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

# Simpan model
joblib.dump(model, "random_forest_model.3.pkl")
print("Model berhasil disimpan sebagai random_forest_model.3.pkl")

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Definisikan rentang jumlah estimator
param_grid = {'n_estimators': [100, 200, 300, 500]}

# Grid Search untuk mencari nilai terbaik
grid_search = GridSearchCV(RandomForestClassifier(random_state=42),
                           param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# Latih model dengan Grid Search
grid_search.fit(X_train, y_train)

# Cetak nilai terbaik
print(f"Best n_estimators: {grid_search.best_params_['n_estimators']}")

from sklearn.model_selection import GridSearchCV

# Definisikan rentang max_depth yang ingin diuji
param_grid = {'max_depth': [5, 10, 15, 20, None]}

# Grid Search untuk mencari nilai terbaik
grid_search = GridSearchCV(RandomForestClassifier(n_estimators=1000, random_state=42),
                           param_grid, cv=5, scoring='accuracy')

# Latih model dengan Grid Search
grid_search.fit(X_train, y_train)

# Cetak nilai max_depth terbaik
print(f"Best max_depth: {grid_search.best_params_['max_depth']}")

# Evaluasi model terbaik
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Optimized Model Accuracy: {accuracy:.2f}")

import datetime
import re
import socket
import requests
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import whois

def extract_features(url):
    parsed_url = urlparse(url)
    domain = parsed_url.netloc
    path = parsed_url.path
    
    features = {}
    
    # 1. HTTPS (1 jika HTTPS, 0 jika HTTP)
    features['HTTPS'] = 1 if parsed_url.scheme == 'https' else 0
    
    # 2. PrefixSuffix (- dalam domain)
    features['PrefixSuffix'] = 1 if '-' in domain else 0
    
    # 3. SubDomains (jumlah subdomain)
    features['SubDomains'] = len(domain.split('.')) - 2 if domain.count('.') > 1 else 0
    
    # 4. RequestURL (proporsi sumber daya eksternal)
    try:
        response = requests.get(url, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')
        total_links = len(soup.find_all('img') + soup.find_all('script') + soup.find_all('link'))
        external_links = sum(1 for tag in soup.find_all(['img', 'script', 'link']) if tag.has_attr('src') and urlparse(tag['src']).netloc != domain)
        features['RequestURL'] = external_links / total_links if total_links > 0 else 0
    except:
        features['RequestURL'] = 0
    
    # 5. AnchorURL (persentase anchor link eksternal)
    try:
        total_anchors = len(soup.find_all('a'))
        external_anchors = sum(1 for tag in soup.find_all('a') if tag.has_attr('href') and urlparse(tag['href']).netloc != domain)
        features['AnchorURL'] = external_anchors / total_anchors if total_anchors > 0 else 0
    except:
        features['AnchorURL'] = 0
    
    # 6. Redirecting// (redirect count dalam URL)
    features['Redirecting//'] = url.count('//') - 1
    
    # 7. UsingIP (1 jika domain adalah IP, 0 jika bukan)
    features['UsingIP'] = 1 if re.match(r'\d+\.\d+\.\d+\.\d+', domain) else 0
    
    # 8. LongURL (1 jika panjang URL > 75 karakter)
    features['LongURL'] = 1 if len(url) > 75 else 0
    
    # 9. ShortURL (1 jika menggunakan shortener)
    shorteners = ['bit.ly', 'tinyurl', 'goo.gl', 't.co', 'ow.ly', 'is.gd']
    features['ShortURL'] = 1 if any(short in domain for short in shorteners) else 0
    
    # 10. DomainRegLen (panjang domain, dihitung berdasarkan panjang domain utama)
    features['DomainRegLen'] = len(domain)
    
    # 11. Favicon (1 jika favicon dari domain berbeda)
    try:
        favicon_link = soup.find('link', rel='shortcut icon')
        if favicon_link and 'href' in favicon_link.attrs:
            features['Favicon'] = 1 if urlparse(favicon_link['href']).netloc != domain else 0
        else:
            features['Favicon'] = 0
    except:
        features['Favicon'] = 0
    
    # 12. NonStdPort (1 jika menggunakan port non-standar)
    features['NonStdPort'] = 1 if ':' in domain and domain.split(':')[-1] not in ['80', '443'] else 0

    # 13. ServerFormHandler (1 jika form mengarah ke domain berbeda)
    try:
        total_forms = len(soup.find_all('form'))
        external_forms = sum(1 for form in soup.find_all('form') if form.has_attr('action') and urlparse(form['action']).netloc != domain)
        features['ServerFormHandler'] = external_forms / total_forms if total_forms > 0 else 0
    except:
        features['ServerFormHandler'] = 0
    
    # 14. WebsiteTraffic (dummy nilai 1 jika domain terkenal, 0 jika tidak, bisa diganti dengan API traffic)
    features['WebsiteTraffic'] = 1 if domain.endswith(('.com', '.org', '.net')) else 0
    
    # 15. GoogleIndex (1 jika terindeks Google, 0 jika tidak, perlu API untuk akurat)
    features['GoogleIndex'] = 1
    
    # 16. LinksInScriptTags (jumlah link dalam script/meta/style)
    try:
        features['LinksInScriptTags'] = sum(1 for tag in soup.find_all(['script', 'meta', 'style']) if tag.has_attr('src'))
    except:
        features['LinksInScriptTags'] = 0
    
    # 17. AbnormalURL (1 jika URL tidak sesuai dengan domain yang diklaim)
    features['AbnormalURL'] = 1 if domain not in url else 0
    
    # 18. WebsiteForwarding (dummy nilai, bisa diambil dari history navigasi)
    features['WebsiteForwarding'] = 0
    
    # 19. InfoEmail (1 jika ada email di halaman)
    features['InfoEmail'] = 1 if re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', response.text) else 0
    
    # 20. UsingPopupWindow (1 jika ditemukan popup window)
    features['UsingPopupWindow'] = 1 if 'window.open' in response.text else 0
    
    # 21. StatusBarCust (1 jika ada manipulasi status bar)
    features['StatusBarCust'] = 1 if 'status=' in response.text else 0
    
    # 22. IframeRedirection (1 jika ada iframe mencurigakan)
    features['IframeRedirection'] = 1 if '<iframe' in response.text else 0
    
    # 23. DisableRightClick (1 jika klik kanan dinonaktifkan)
    features['DisableRightClick'] = 1 if 'event.button==2' in response.text else 0
    
    # 24. Symbol@ (1 jika URL mengandung '@')
    features['Symbol@'] = 1 if '@' in url else 0
    
    # 25. HTTPSDomainURL (1 jika domain utama menggunakan HTTPS)
    features['HTTPSDomainURL'] = features['HTTPS']

    # 26. DNSRecording (1 jika DNS record ditemukan, 0 jika tidak)
    try:
        domain_info = whois.whois(domain)
        features['DNSRecording'] = 1 if domain_info else 0
    except:
        features['DNSRecording'] = 0

    # 27. AgeofDomain (umur domain dalam hari)
    try:
        domain_info = whois.whois(domain)
        creation_date = domain_info.creation_date if domain_info and domain_info.creation_date else None
        if isinstance(creation_date, list):
            creation_date = creation_date[0]
        features['AgeofDomain'] = (datetime.now() - creation_date).days if creation_date else 0
    except:
        features['AgeofDomain'] = 0

    # 28. PageRank (dummy nilai, bisa diambil dari API)
    features['PageRank'] = 0.5

    # 29. LinksPointingToPage (jumlah link yang mengarah ke halaman)
    try:
        features['LinksPointingToPage'] = len(soup.find_all('a', href=True))
    except:
        features['LinksPointingToPage'] = 0
    
    return features


# URL baru yang ingin diuji
url_test = "http://exemple.gistemp.com/mail/en.php"

# Ekstraksi fitur dari URL
features = extract_features(url_test)

# Prediksi menggunakan model
prediction = model.predict(X_test)

# Interpretasi hasil prediksi
if prediction[0] == -1:
    print("URL ini adalah phishing")
else:
    print("URL ini aman")