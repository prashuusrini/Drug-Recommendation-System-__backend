from flask import Flask, request, jsonify
from your_ml_model import predict_drugs  # Your custom function

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    drugs = predict_drugs(data)  # Your ML prediction logic
    return jsonify({"recommendations": drugs})

if __name__ == "__main__":
    app.run(debug=True)
