import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import joblib

# -------------------------
# Step 1: Sample Training Data
# -------------------------
data = {
    'symptoms': [
        'fever cough headache',
        'chest pain high blood pressure',
        'fatigue weight loss high blood sugar',
        'rash itching sneezing',
        'joint pain stiffness',
        'abdominal pain nausea vomiting'
    ],
    'drug_1': [1, 0, 0, 1, 0, 0],
    'drug_2': [0, 1, 0, 0, 0, 1],
    'drug_3': [0, 0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

X = df['symptoms']
y = df[['drug_1', 'drug_2', 'drug_3']]

# -------------------------
# Step 2: Build Pipeline
# -------------------------
pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultiOutputClassifier(RandomForestClassifier()))
])

# -------------------------
# Step 3: Train Model
# -------------------------
pipeline.fit(X, y)

# -------------------------
# Step 4: Save Model
# -------------------------
joblib.dump(pipeline, 'model.pkl')

print("✅ model.pkl has been created.")
