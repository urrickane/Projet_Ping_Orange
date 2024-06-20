# Borne Metahuman Orange - IA Cloud

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)

## Description

Ce dossier contient la partie Intelligence Artificielle de l'application. Elle n'est composée que d'un script à lancer afin de tirer et de lancer l'image Docker contenant l'IA. Cette dernière est disponible sur le cloud Microsoft Azure.
En cas d'erreurs de politiques CORS lors de l'exécution de requêtes sur un navigateur ou de problèmes avec le service Azure de Microsoft, nous vous conseillons d'utiliser plutôt l'IA en local dans le dossier "Local" situé dans /IA/Local.

## Installation et utilisation

### Prérequis

[Docker Desktop](https://www.docker.com/products/docker-desktop/)

### Instructions

1. Lancer le logiciel Docker Desktop

2. Naviguer dans le répertoire du dépôt GitHub.

```bash
cd Projet_Ping_Orange/IA/Cloud
```

3. Lancer le script pour démarrer le serveur local.

```bash
bash start_cloud_server.sh
```

### Remarques

Nous conseillons d'attendre quelques instants le temps que le serveur démarre correctement. Pour les machines les plus lentes, il arrive que les premières requêtes résultent en un échec.
