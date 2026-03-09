# Étude du chômage en France - Analyse et prédictions (Projet Fil Rouge MS EBDE)
Ce projet de data science vise à modéliser et prédire les dynamiques du marché du travail français. Nous croisons les données de l'INSEE et de la DARES avec des indicateurs macroéconomiques (PIB, Inflation, Climat des affaires) pour comprendre les cycles du chômage. Ce dépôt contient l'intégralité du pipeline : du nettoyage des données à la modélisation prédictive via des approches linéaires (Ridge) et non-linéaires (LSTM, Random Forest).

## Objectifs du projet
- Acquisition : Centralisation de données multi-sources (Mensuelles, Trimestrielles, Annuelles).
- Exploration : Corrélation entre indicateurs macroéconomiques (PIB, intérim, inflation) et le taux de chômage INSEE.
- Feature Engineering : Trimestrialisation des données, stationnarisation (taux de croissance), et création de signaux avancés (lags, Z-scores).
- Feature Selection : Utilisation de Lasso pour identifier les variables réellement motrices.
- Analyse : Étude de la Loi d'Okun, la courbe de Phillips et des impacts démographiques (âge, sexe).
- Benchmarking : Comparaison de modèles (Ridge, Random Forest, XGBoost, LSTM).
- Forecasting : Simulation de 3 scénarios pour 2026 (Optimiste, Middle, Pessimiste).

## Installation
### 1. Cloner le projet
```bash
git clone https://github.com/JulienSchnitzler/fil_rouge-etude_chomage_france.git
cd fil_rouge-etude_chomage_france
```

### 2. Configurer l'environnement
#### Via Pip (standard)
Sur Windows :
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
Sur Mac / Linux :
```bash
python3 -m venv venv
source .venv/bin/activate
pip install -r requirements.txt
```
Lancez Jupyter Lab  :
```bash
jupyter lab
```
Puis aller dans le dossier `src` (data/features/models) pour retrouver les différents notebooks.

#### Via UV (recommandé, plus rapide !)
Si vous utilisez [UV](https://github.com/astral-sh/uv), l'installation est quasi instantanée :
```bash
uv sync
```
Puis 
```bash
# Sur Mac / Linux :
source .venv/bin/activate

# Sur Windows (PowerShell) :
.venv\Scripts\activate
```
Lancer le projet
```bash
uv run jupyter lab
```
Puis aller dans le dossier `src` (data/features/models) pour retrouver les différents notebooks.

#### Via Anaconda / Conda
Idéal pour les environnements de Data Science isolés :
```bash
conda create --name chomage_env python=3.10
conda activate chomage_env
conda install --file requirements.txt

# Ou via le fichier environment.yml :
conda env create -f environment.yml
```
Puis lancer le projet
```bash
jupyter lab
```

## Technologies
- Data : Pandas, Numpy.
- Visualisation : Matplotlib, Seaborn.
- Machine Learning : Scikit-Learn (LassoCV, Ridge, Random Forest).
- Deep Learning : TensorFlow / Keras (LSTM).

## Structure du projet
```plaintext
.
├── data/   
│   ├── processed/              # Datasets nettoyés prêts pour l'analyse
│   ├── 01_dataset_bronze.csv   # Jointure des données sources
│   └── 02_dataset_silver.csv   # Données nettoyées et transformées
│   └── raw/                    # Données brutes (fichiers sources .csv, .xlsx etc.)
├── docs/                       # Référentiel (COG Insee, dictionnaire des variables)
├── notebooks/                  # Notebooks Jupyter pour les analyses
├── reports/
│   └── presentations/          # PowerPoints des présentations mensuelles
├── src/                        # Code source Python (scripts de nettoyage/modèles)
│   ├── data/                   # Ajout des variables exogènes
│   ├── features/               # Création de features
│   └── models/                 # Modélisation
├── .gitignore                  # Fichiers à exclure (venv, .ipynb_checkpoints, etc.)
├── environment.yml             # Configuration de l'environnement Conda
└── README.md                   # Ce fichier
```

## Workflow de travail
1. Données : Toute nouvelle donnée brute doit être documentée dans le fichier `docs/referentiel_sources_chomage.xlsx`.
2. Code : Ne travaillez jamais dans la branche main directement. Créez une branche par fonctionnalité ou par étape.
3. Chemins : Utilisez toujours la librairie pathlib pour que vos chemins fonctionnent sur Linux, Mac et Windows.
