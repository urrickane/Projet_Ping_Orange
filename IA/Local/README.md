# Borne Metahuman Orange - IA Locale

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)

## Description

Ce dossier contient la partie Intelligence Artificielle de l'application. Elle est composée de plusieurs fichiers et d'un seul script à lancer afin de démarrer le serveur en local sur localhost:5000/.
Il n'est conseillé d'utiliser cette version qu'en cas d'erreurs de politiques CORS sur un navigateur, de pannes du service Azure de Microsoft ou dans l'optique de modifier l'IA de manière à la perfectionner.

## Installation et utilisation

### Prérequis

[Python 3.11](https://www.python.org/downloads/release/python-3110/)

### Instructions

1. Naviguer dans le répertoire du dépôt GitHub.

```bash
cd <votre-repo>/IA/Local
```

2. Lancer le script pour démarrer le serveur local.

```bash
bash start_local_server.sh
```

### Remarques

Nous conseillons d'attendre quelques instants le temps que le serveur démarre correctement. Pour les machines les plus lentes, il arrive que les premières requêtes résultent en un échec.

--- 

## Détails des Fichiers

### start_local_server.sh

Le script bash pour démarrer le serveur local Flask. Il crée et active un environnement virtuel, installe les dépendances répertoriées dans `requirements.txt`, télécharge le modèle de langue spaCy pour le français, puis lance `app.py`.

### app.py

Ce fichier contient le code de l'API Flask. Il définit un endpoint `/predict` qui accepte des requêtes POST pour prédire la catégorie d'une phrase donnée. Le modèle de machine learning et le vocabulaire sont chargés au démarrage du serveur.

### best_model.pt

Ce fichier contient le modèle de machine learning pré-entraîné. Il est utilisé par l'API Flask pour faire des prédictions sur les phrases fournies via les requêtes POST.

### vocab.pkl

Ce fichier contient le vocabulaire utilisé par le modèle de machine learning pour tokeniser les phrases d'entrée. Il est chargé au démarrage de l'API Flask pour assurer une tokenisation cohérente avec celle utilisée lors de l'entraînement du modèle.
