from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("ensemble.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl","rb"))
encoders = pickle.load(open("encoders.pkl","rb"))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    raw = request.json  # All 16 features in the same order as training

    # The order must match exactly how it was transformed during training
    feature_order = [
        "Gender", "family_history_with_overweight", "FAVC", "CAEC", "SMOKE",
        "SCC", "CALC", "MTRANS", "Age", "Height", "Weight", "FCVC", "NCP",
        "CH2O", "FAF", "TUE"
    ]

    # Apply label encoding for categorical fields
    cat_features = ["Gender", "family_history_with_overweight", "FAVC",
                    "CAEC", "SMOKE", "SCC", "CALC", "MTRANS"]
    for c in cat_features:
        raw[c] = encoders[c].transform([raw[c]])[0]

    # Build one full feature row (16 columns)
    row = [raw[f] for f in feature_order]

    # Scale all 16 features at once
    scaled_row = scaler.transform([row])[0]

    # Predict
    prediction = model.predict([scaled_row])[0]
    return jsonify({"prediction": prediction})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

