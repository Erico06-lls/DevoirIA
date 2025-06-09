# 🏠 Prédiction de prix de maisons avec un modèle de régression (Keras)

Ce petit projet Python permet de prédire le prix de maisons en utilisant un modèle de réseau de neurones (**Dense**) de la bibliothèque **TensorFlow/Keras**.

## 🔷 Fonctionnalités

✅ Génération d’un fichier CSV (`maisons.csv`) avec des caractéristiques réalistes de maisons :  
- Surface (m²)  
- Nombre de chambres  
- Quartier  
- État du bâtiment (0 ou 1)  
- Prix réel

✅ Lecture et prétraitement des données (normalisation, séparation train/test)

✅ Création d’un **réseau de neurones** avec 6 perceptrons et une couche de sortie pour la **régression**

✅ Entraînement, évaluation et sauvegarde du modèle (`modele_maisons.keras`)

✅ Rechargement du modèle pour de futures prédictions

## 🚀 Pour exécuter

1. Installe les dépendances (si nécessaire) :

bash:
pip install pandas numpy scikit-learn tensorflow

2. Lance le script Python principal :

bash:
python main.py

3. Vérifie la sortie en console :

* MSE (Mean Squared Error)
* MAE (Mean Absolute Error)
* 5 premières prédictions du prix de maisons

## 📊 Précision / performance

Comme il s’agit d’un problème de **régression**, on mesure l’erreur moyenne (MAE) plutôt qu’un « pourcentage de précision ».

* Le modèle affiche en console la **MSE** (erreur quadratique moyenne) et la **MAE** (erreur absolue moyenne).
* Dans nos tests, sur les données générées aléatoirement, la MAE est généralement de l’ordre de **20 000 à 30 000**.

ℹ️ Cela représente environ **10–15 % d’erreur moyenne** par rapport au prix réel (variable selon la complexité et la distribution des données).

## 📁 Structure des fichiers

├── main.py       # Script principal
├── maisons.csv
├── modele_maisons.keras
├── README.md
├── modele.py
└── CSV.py

## 💡 Personnalisation

Pour un fichier CSV réel, il suffit de remplacer la ligne :

df = pd.read_csv('maisons.csv')
par :

df = pd.read_csv('ton_fichier_reel.csv')
en veillant à ce que les colonnes correspondent.

TOLOJANAHARY Jean Erico