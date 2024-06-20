# Utiliser une image de base Python
FROM python:3.8-slim

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier les fichiers de l'application dans le conteneur
COPY . /app

# Installer les dépendances nécessaires
RUN pip install --no-cache-dir torch flask flask-cors torchtext spacy
RUN python -m spacy download fr_core_news_sm

# Exposer le port sur lequel l'application va tourner
EXPOSE 5000

# Commande pour démarrer l'application
CMD ["python", "app.py"]
