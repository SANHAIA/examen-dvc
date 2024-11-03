import json
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, r2_score

# Charger le modèle et les ensembles de test
model = joblib.load('models/trained_model.pkl')
X_test = pd.read_csv('data/processed/X_test_scaled.csv')
y_test = pd.read_csv('data/processed/y_test.csv')

# Prédictions
predictions = model.predict(X_test)
pd.DataFrame(predictions, columns=['predictions']).to_csv('data/processed/predictions.csv', index=False)

# Évaluation des performances
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
metrics = {'mse': mse, 'r2': r2}

# Sauvegarder les métriques
with open('metrics/scores.json', 'w') as f:
    json.dump(metrics, f)
