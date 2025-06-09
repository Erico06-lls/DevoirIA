import pandas as pd
import numpy as np

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