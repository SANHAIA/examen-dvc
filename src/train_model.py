import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor

# Charger les données normalisées et les meilleurs paramètres
X_train = pd.read_csv('data/processed/X_train_scaled.csv')
y_train = pd.read_csv('data/processed/y_train.csv')
best_params = joblib.load('models/best_params.pkl')

# Entraîner le modèle
model = RandomForestRegressor(**best_params, random_state=42)
model.fit(X_train, y_train.values.ravel())

# Sauvegarder le modèle entraîné
joblib.dump(model, 'models/trained_model.pkl')
