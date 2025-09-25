import os
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load("model.pkl")  # Load your model

@app.route("/")
def home():
    return "Drug Recommendation API is running."

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    # Custom prediction logic here
    recommendations = ["Paracetamol", "Ibuprofen"]  # Dummy response
    return jsonify({"recommendations": recommendations})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
