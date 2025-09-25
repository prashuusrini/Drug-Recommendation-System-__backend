import joblib

model = joblib.load("model.pkl")

def predict_drugs(symptom_input: str):
    preds = model.predict([symptom_input])[0]
    drug_labels = ['Paracetamol', 'Atenolol', 'Metformin']  # Map to real drug names
    return [drug for drug, p in zip(drug_labels, preds) if p == 1]
