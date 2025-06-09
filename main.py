import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential, load_model
from keras.layers import Dense

# creer et sauvegarder un fichier CSV simule
np.random.seed(42)
n_samples = 50

surface_m2 = np.random.randint(50, 200, size=n_samples)
chambres = np.random.randint(1, 6, size=n_samples)
quartier = np.random.randint(1, 4, size=n_samples)
etat_batiment = np.random.randint(0, 2, size=n_samples)

prix = (surface_m2 * 2500) + (chambres * 1000) + (quartier * 5000) + (etat_batiment * 20000) + np.random.randint(-20000, 20000, size=n_samples)

df = pd.DataFrame({
    'surface_m2' : surface_m2,
    'chambres' : chambres,
    'quartier' : quartier,
    'etat_batiment' : etat_batiment,
    'prix': prix
})

df.to_csv('maison.csv', index=False)
print("Fichier 'maison.csv' cree avec succes !")
print(df.head())

# Charger et preparer les donnees
df = pd.read_csv('maison.csv')
X = df[['surface_m2', 'chambres', 'quartier', 'etat_batiment']]
y = df['prix']

# Normaliser les donnees
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Separer en train train/test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Creer et compiler le modele
model = Sequential([
    Dense(6, activation = 'relu', input_shape=(X_train.shape[1],)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'],)

# Entrainer le modele
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Evaluer et predire
loss, mae = model.evaluate(X_test, y_test)
print(f"Test MSE : {loss:.2f}, Test MAE : {mae:.2f}")

predictions = model.predict(X_test)
print("5 premiere predictions : ", predictions[:5].flatten())

# Sauvegarder le modele
model.save('modele_maisons.keras')
print("Modele sauvegarde sous modele_maisons.keras")

# Recharger le modele et verifier
modele_charge = load_model('modele_maisons.keras')
preds_reload = modele_charge.predict(X_test)
print("Verification : prediction rechargees (5 premieres) :", preds_reload[:5].flatten())