# Crop Type Classification — Notebook Explanation

##  Description
Ce dépôt contient un notebook (projet ML.ipynb) qui implémente un pipeline complet pour la classification du type de culture à partir de paramètres pédologiques et météorologiques.

##  Fichier de données
- Fichier attendu : `Crop_recommendation.csv` (séparateur `;`).
- Colonnes principales : `N`, `P`, `K`, `temperature`, `humidity`, `ph`, `rainfall`, `label`.

##  Aperçu du notebook (sections)
1. **Import & configuration** — import des bibliothèques et configuration des graphiques.
2. **Chargement des données** — lecture du CSV et vérification basique (shape, head, info).
3. **EDA** — statistiques descriptives, contrôle des valeurs manquantes et doublons, distribution de la variable cible.
4. **Visualisation** — histogrammes, heatmap de corrélation, boxplots pour détecter les outliers.
5. **Feature engineering** — création de ratios nutritifs (ex : `N_P_ratio`, `N_K_ratio`, `P_K_ratio`).
6. **PCA** — réduction de dimension pour visualisation (2 composantes principales).
7. **Préparation ML** — encodage de la cible, split train/test, mise à l'échelle.
8. **Entraînement & Évaluation** — tests de plusieurs modèles (Logistic Regression, k-NN, SVM, Random Forest) et affichage des métriques + matrices de confusion.
9. **Optimisation d'hyperparamètres** — exemple de GridSearch pour Random Forest (optionnel, peut être long).
10. **Interprétabilité** — importance des caractéristiques (feature importances pour Random Forest).
11. **Export** — sauvegarde du modèle, du scaler et des classes du label encoder (`random_forest_model.pkl`, `scaler.pkl`, `label_encoder.pkl`).

## ⚙️ Exécution
- Assurez-vous d'avoir Python 3.8+ et les paquets suivants : `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`.
- Exemple d'installation :

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

- Placez `Crop_recommendation.csv` dans le même répertoire que le notebook puis ouvrez et exécutez les cellules dans l'ordre.

---

# English README — Notebook explanation

## Description
This repository contains a notebook (`projet ML.ipynb`) that implements a complete pipeline for crop type classification from soil and weather parameters.

## Dataset
- Expected file: `Crop_recommendation.csv` (separator `;`).
- Main columns: `N`, `P`, `K`, `temperature`, `humidity`, `ph`, `rainfall`, `label`.

## Notebook overview (sections)
1. **Import & configuration** — import libraries and plotting configuration.
2. **Data loading** — read the CSV and basic checks (shape, head, info).
3. **EDA** — descriptive statistics, checking missing values and duplicates, target distribution.
4. **Visualization** — histograms, correlation heatmap, boxplots for outlier detection.
5. **Feature engineering** — creation of nutrient ratios (e.g., `N_P_ratio`, `N_K_ratio`, `P_K_ratio`).
6. **PCA** — dimensionality reduction for visualization (2 principal components).
7. **ML preparation** — label encoding, train/test split, scaling.
8. **Training & Evaluation** — testing several models (Logistic Regression, k-NN, SVM, Random Forest) and showing metrics + confusion matrices.
9. **Hyperparameter tuning** — example GridSearch for Random Forest (optional; can be time-consuming).
10. **Interpretability** — feature importance (Random Forest feature importances).
11. **Export** — saving the model, the scaler and the label encoder classes (`random_forest_model.pkl`, `scaler.pkl`, `label_encoder.pkl`).

## Execution
- Make sure you have Python 3.8+ and the following packages: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`.
- Installation example:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

- Place `Crop_recommendation.csv` in the same directory as the notebook and run the cells in order.