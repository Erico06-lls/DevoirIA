from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Charger les donnees
housing = fetch_california_housing()
X = housing.data
y = housing.target

# Diviser en train et test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)

# Normaliser les donnees
scaler = StandardScaler()
X_train = scaler.fit_transform(X_test)
X_test = scaler.fit_transform(X_test)

# Creer le modele
model = Sequential()
model.add(Dense(6, activation = 'relu', input_shape=(X_train.shape[1],)))
model.add(Dense(1))

# Compiler le modele
model.compile(optimizer='adam', loss='mse', metrics=['mae'],)

# Entrainnement du modele
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Resume
model.summary()

# Visualiser les performance de la courbe
plt.plot(history.history['loss'], label='Loss (entraînement)')
plt.plot(history.history['val_loss'], label='Loss (validation)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(history.history['mae'], label='MAE (entraînement)')
plt.plot(history.history['val_mae'], label='MAE (validation)')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()
plt.show()

# Evaluer le modele sur les donnees de test
loss, mae = model.evaluate(X_test, y_test, verbose=1)
print(f"Test loss (MSE) : {loss:.4f}")
print(f"Test MAE : {mae:.4f}")

# Exemple de prediction sur les donnees de test
predictions = model.predict(X_test)
print(predictions[:5])