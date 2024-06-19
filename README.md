# Projet_Ping_Orange

Projet Ping Orange FISE2

## Description

Le projet Ping Orange vise à créer un métahuman pour une borne interactive dans un magasin d'un opérateur téléphonique comme Orange. L'idée est que cette borne puisse accueillir les clients et comprendre leurs demandes grâce à une intelligence artificielle. Le métahuman, conçu dans Unreal Engine 5, sera affiché dans un navigateur web via le pixel streaming.

Le système fonctionne de la manière suivante :
1. **Interaction avec le client** : Le client interagit avec la borne en formulant une demande vocale ou textuelle.
2. **Traitement de la demande** : La demande est envoyée à une API Flask qui utilise un modèle d'IA pour catégoriser la demande en différentes catégories (par exemple, "Achat", "Rendez-vous", "Autres").
3. **Exécution du scénario** : En fonction de la catégorie de la demande, un scénario spécifique est déclenché dans Unreal Engine avec le métahuman qui guide le client.

## Architecture du Projet

Le projet est organisé en trois répertoires principaux :

- **UnrealEngine** : Contient le projet Unreal Engine 5 avec le métahuman.
- **IA** : Contient le modèle d'IA et l'API Flask pour traiter et catégoriser les demandes des clients.
    - **Cloud** : Contient la solution pour la partie Cloud, incluant une image Docker à télécharger.
    - **Local** : Contient la solution locale.
- **PixelStreaming** : Contient les scripts et configurations nécessaires pour lancer le pixel streaming.

### Schéma de l'Architecture

![Project Outline](./Project_Outline.png)

## Prérequis

Avant de lancer le projet, assurez-vous d'avoir les éléments suivants installés :

- **Unreal Engine 5**
- **Python 3.x**
- **Flask**
- **PyTorch**
- **Spacy avec le modèle français (`fr_core_news_sm`)**
- **Navigateur compatible WebRTC**

## Installation

1. **Cloner le dépôt** :
    ```bash
    git clone https://github.com/yourusername/Projet_Ping_Orange.git
    cd Projet_Ping_Orange
    ```

2. **Configurer l'environnement Python** :
    ```bash
    pip install -r IA/requirements.txt
    python -m spacy download fr_core_news_sm
    ```

3. **Lancer l'API Flask** :
    - Pour exécuter la solution Cloud, naviguez dans le sous-dossier `IA/Cloud` et exécutez le script :
        ```bash
        cd IA/Cloud
        ./start_cloud_server.sh
        ```
    - Pour exécuter la solution locale, naviguez dans le sous-dossier `IA/Local` et exécutez le script :
        ```bash
        cd IA/Local
        ./start_local_server.sh
        ```

4. **Ouvrir le projet Unreal Engine** :
    - Ouvrez Unreal Engine 5 et chargez le projet dans le répertoire `UnrealEngine`.

5. **Démarrer le Pixel Streaming** :
    - Suivez les instructions dans le répertoire `PixelStreaming` pour démarrer le pixel streaming.

## Contenu des Répertoires

- **UnrealEngine** : Ce répertoire contient le projet Unreal Engine 5. Il inclut tous les fichiers nécessaires pour le métahuman et les scénarios d'interaction.
  
- **IA** : Ce répertoire contient deux sous-dossiers :
    - **Cloud** : Contient une solution basée sur le cloud avec une image Docker préconfigurée. Vous pouvez exécuter `start_cloud_server.sh` pour démarrer le serveur sur le cloud.
    - **Local** : Contient une solution locale pour l'IA. Vous pouvez exécuter `start_local_server.sh` pour démarrer le serveur localement.

- **PixelStreaming** : Ce répertoire contient les scripts nécessaires pour configurer et lancer le pixel streaming. Cela permet de diffuser le contenu d'Unreal Engine 5 dans un navigateur web.

## Contributeurs

- [urrickane](https://github.com/urrickane)

## License

Ce projet est sous licence MIT - voir le fichier [LICENSE](LICENSE) pour plus de détails.

