# Analyse du Chômage en France - Projet Fil Rouge

Ce projet vise à modéliser et prédire les dynamiques du marché du travail français. Nous croisons les données de l'INSEE et de la DARES avec des indicateurs macroéconomiques (PIB, Inflation, Climat des affaires) pour comprendre les cycles du chômage.

## Objectifs du projet

- Acquisition : Centralisation de données multi-sources (Mensuelles, Trimestrielles, Annuelles).
- Feature Engineering : Trimestrialisation des données, stationnarisation (taux de croissance), et création de signaux avancés (lags, Z-scores).
- Analyse : Étude de la Loi d'Okun et des impacts démographiques (âge, sexe).
P- rédiction : Modélisation statistique et Machine Learning.

## Installation rapide

Pour que nous ayons tous exactement la même configuration (versions de Python et des librairies), nous utilisons **Conda**.

### 1. Cloner le projet
```bash
git clone https://github.com/JulienSchnitzler/fil_rouge-etude_chomage_france
cd fil_rouge-etude_chomage_france
```

### 2. Si vous utilisez pip ...

...

### 2. Si vous utilisez uv ...

...

### 2. Si vous utilisez Conda ...

#### a. Créer l'environnement virtuel
Utilise le fichier `environment.yml` fourni pour recréer l'environnement :
```bash
conda env create -f environment.yml
```

#### b. Activer l'environnement
```bash
conda activate env_chomage
```

### c. Lier l'environnement à Jupyter
```bash
python -m ipykernel install --user --name env_chomage --display-name "Python (Projet Chomage)"
```

## Structure du projet
```plaintext
.
├── data/   
│   ├── processed/      # Datasets nettoyés prêts pour l'analyse
│   └── raw/            # Données brutes (ignorées par Git si > 100Mo)
├── docs/               # Référentiel (COG Insee, dictionnaire des variables)
├── notebooks/          # Notebooks Jupyter pour les analyses
├── reports/
│   └── presentations/  # PowerPoints des présentations mensuelles
├── src/                # Code source Python (scripts de nettoyage/modèles)
│   ├── data/           # Ajout des variables exogènes
│   ├── features/       # Création de features
│   └── models/         # Modélisation
├── environment.yml     # Configuration de l'environnement Conda
└── README.md           # Ce fichier
```

## Workflow de travail
1. Données : Toute nouvelle donnée brute doit être documentée dans le fichier `data/external/referentiel_sources_chomage.xlsx`.
2. Code : Ne travaillez jamais dans la branche main directement. Créez une branche par fonctionnalité ou par étape.
3. Chemins : Utilisez toujours la librairie pathlib pour que vos chemins fonctionnent sur Windows et Mac.