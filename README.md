# Analyse du Chômage en France - Projet Fil Rouge

Ce projet vise à analyser les dynamiques du marché du travail français en utilisant les données de l'INSEE et de la DARES. Il couvre l'acquisition, le nettoyage, l'analyse exploratoire et la modélisation des taux de chômage.

## Installation rapide

Pour que nous ayons tous exactement la même configuration (versions de Python et des librairies), nous utilisons **Conda**.

### 1. Cloner le projet
```bash
git clone https://github.com/JulienSchnitzler/fil_rouge-etude_chomage_france
cd ... A RAJOUTER ...
```

### 2. Créer l'environnement virtuel
Utilise le fichier `environment.yml` fourni pour recréer l'environnement :
```bash
conda env create -f environment.yml
```

### 3. Activer l'environnement
```bash
conda activate env_chomage
```

### 4. Lier l'environnement à Jupyter
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