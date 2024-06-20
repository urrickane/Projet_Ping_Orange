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
cd Projet_Ping_Orange/IA/Local
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

Le script bash à lancer pour démarrer le serveur local Flask. Il crée et active un environnement virtuel, installe les dépendances répertoriées dans `requirements.txt`, télécharge le modèle de langue spaCy pour le français, puis lance `app.py`.

### app.py

Ce fichier contient le code de l'API Flask. Il définit un endpoint `/predict` qui accepte des requêtes POST pour prédire la catégorie d'une phrase donnée. Le modèle de machine learning et le vocabulaire sont chargés au démarrage du serveur.

### best_model.pt

Ce fichier contient le modèle de machine learning pré-entraîné. Il est utilisé par l'API Flask pour faire des prédictions sur les phrases fournies via les requêtes POST.

### vocab.pkl

Ce fichier contient le vocabulaire utilisé par le modèle de machine learning pour tokeniser les phrases d'entrée. Il est chargé au démarrage de l'API Flask pour assurer une tokenisation cohérente avec celle utilisée lors de l'entraînement du modèle.

### datav3.csv

Ce fichier contient le jeu de données utilisé pour entraîner le modèle de machine learning. Il peut être modifié pour ajouter de nouvelles données ou de nouvelles catégories afin d'améliorer ou d'étendre les capacités du modèle.

### train_model.py

Ce script Python permet d'entraîner le modèle de machine learning. Il utilise les données présentes dans `datav3.csv` et génère un nouveau fichier `best_model.pt` et `vocab.pkl`. Il est utile pour ajuster les paramètres de l'IA ou pour inclure de nouvelles données d'entraînement.

### Dockerfile

Ce fichier est utilisé pour créer une image Docker du serveur Flask avec l'IA intégrée. Il définit l'environnement d'exécution et les dépendances nécessaires pour faire tourner l'application dans un conteneur Docker.

---

## Remarque

Tous ces fichiers et scripts sont mis à disposition pour permettre à l'utilisateur de modifier, personnaliser ou étendre ces travaux, afin de les adapter précisément à ses besoins spécifiques et d'améliorer les performances de l'IA selon ses propres critères et exigences.
