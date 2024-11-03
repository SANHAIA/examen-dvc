import pandas as pd
from sklearn.preprocessing import StandardScaler

# Charger les ensembles de données d'entraînement et de test
X_train = pd.read_csv('data/processed/X_train.csv')
X_test = pd.read_csv('data/processed/X_test.csv')


# Sélectionner uniquement les colonnes numériques
X_train_numeric = X_train.select_dtypes(include=['float64', 'int64'])
X_test_numeric = X_test.select_dtypes(include=['float64', 'int64'])

# Appliquer le scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_numeric)
X_test_scaled = scaler.transform(X_test_numeric)

# Convertir les tableaux en DataFrames pour sauvegarder facilement
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train_numeric.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test_numeric.columns)


# Sauvegarder les résultats normalisés
X_train_scaled.to_csv('data/processed/X_train_scaled.csv', index = False)
X_test_scaled.to_csv('data/processed/X_test_scaled.csv', index = False)

