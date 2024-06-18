#!/bin/bash

# Définir le nom de l'environnement virtuel
ENV_NAME="venv"

# Créer un environnement virtuel
python3 -m venv $ENV_NAME

# Activer l'environnement virtuel
source $ENV_NAME/bin/activate

# Installer les dépendances à partir de requirements.txt
pip install -r requirements.txt

# Installer le modèle de langue spaCy
python -m spacy download fr_core_news_sm

echo "Environnement virtuel créé avec succès."
echo "Serveur lancé. Appuyez sur Ctrl+C pour le stopper."

# Lancer le fichier app.py
python app.py
