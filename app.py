from flask import Flask, render_template, request, jsonify
from phising_detector import extract_features
import joblib
import pandas as pd
import os
import urllib.request

app = Flask(__name__)

# Cek dan unduh model jika belum ada
model_filename = "random_forest_model.3.pkl"
model_url = "https://drive.google.com/uc?export=download&id=1hvg7fqxxsYDCKQFQ37kQaqExZHtUnXlz"

if not os.path.exists(model_filename):
    print("📥 Downloading model from Google Drive...")
    urllib.request.urlretrieve(model_url, model_filename)
    print("✅ Model downloaded!")

# Load model
model = joblib.load(model_filename)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/url')
def url_page():
    return render_template('url.html')

@app.route('/scanqr')
def scanqr_page():
    return render_template('scanqr.html')

@app.route('/analyze', methods=['POST'])
def analyze_url():
    try:
        data = request.get_json()
        url_test = data.get('url')

        if not url_test:
            return jsonify({'error': 'URL tidak boleh kosong'}), 400

        features = extract_features(url_test)
        X_input = pd.DataFrame([features])

        expected_features = model.feature_names_in_
        X_input.rename(columns={"PrefixSuffix": "PrefixSuffix-"}, inplace=True)

        if "Index" not in X_input.columns:
            X_input["Index"] = 0

        X_input = X_input.reindex(columns=expected_features, fill_value=0)

        prediction = model.predict(X_input)

        result = {
            'message': "URL tersebut Phishing, Hati-hati, URL ini berbahaya.",
            'color': "red"
        } if prediction[0] == -1 else {
            'message': "URL tersebut bukan Phishing, Anda dapat terus mengakses URL tersebut.",
            'color': "green"
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analyze_qr', methods=['POST'])
def analyze_qr():
    try:
        data = request.get_json()
        qr_url = data.get('qr_url')

        if not qr_url:
            return jsonify({'error': 'QR Code tidak boleh kosong'}), 400

        features = extract_features(qr_url)
        X_input = pd.DataFrame([features])

        expected_features = model.feature_names_in_
        X_input.rename(columns={"PrefixSuffix": "PrefixSuffix-"}, inplace=True)

        if "Index" not in X_input.columns:
            X_input["Index"] = 0

        X_input = X_input.reindex(columns=expected_features, fill_value=0)

        prediction = model.predict(X_input)

        result = {
            'message': "URL tersebut Phishing, Hati-hati, URL ini berbahaya.",
            'color': "red"
        } if prediction[0] == -1 else {
            'message': "URL tersebut bukan Phishing, Anda dapat terus mengakses URL tersebut.",
            'color': "green"
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
