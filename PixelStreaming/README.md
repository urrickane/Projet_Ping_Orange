# Démarrage avec le Pixel Streaming dans Unreal Engine

## Introduction

Le Pixel Streaming permet de diffuser des images haute qualité rendues par Unreal Engine depuis un serveur puissant vers des appareils distants en temps réel.

Dans ce document, vous trouverez un guide pour l’installation et la mise en place de Pixel Streaming, ainsi qu’une description détaillée du premier lancement de stream. De plus, vous découvrirez un script facilitant les futurs démarrages du flux.

## Prérequis

Avant de commencer, assurez-vous d'avoir :
- **Unreal Engine 5.3**.
- Ordinateur compatible, voir les prérequis [UnrealEngine](https://dev.epicgames.com/documentation/fr-fr/unreal-engine/hardware-and-software-specifications-for-unreal-engine).
- [Node.js](https://nodejs.org/en/download/package-manager).
- Ports 80 et 8888, libres et accessibles.

## Installation et Configuration d'Unreal Engine

1. **Téléchargement et Installation** :
   - Téléchargez Unreal Engine depuis le [site officiel](https://www.unrealengine.com/).
   - Installez-le en suivant les instructions fournies.

2. **Activation des Plugins** :
   - Ouvrez votre projet dans Unreal Engine.
   - Allez dans `Edit` -> `Plugins`.
   - Recherchez et activez les plugins `Pixel Streaming`.
   - Puis relancez l'application.

## Configuration du Projet

1. **Paramètres du Projet** :
   - Allez dans `Edit` -> `Editor Preferences`.
   
   - Sous `Level Editor` -> `Play`, recherchez  `Additional Launch Parameters` et entrez 

     `-AudioMixer -PixelStreamingIP=localhost -PixelStreamingPort=8888`.

## Package et Déploiement

1. **Package du Projet** :
   - Allez dans `Platforms` -> `Windows`-> `Package Project`.
   - Suivez les instructions pour packager votre projet.

## **Premier démarrage du Stream** :

1. **Raccourcis application** :

   - Allez dans le dossier crée lors du package puis,  `Windows`.

   - Créez un raccourcis du `.exe` , en maintenant la touche `Alt` et un `"Drag"` de celle-ci.

   - Click droit sur le raccourcis -> `Propriétés` -> Dans `Target`, ajoutez à la fin du chemin un `espace` et collez 

     `-AudioMixer -PixelStreamingIP=localhost -PixelStreamingPort=8888` .

2. **Lancement du Serveur**:

   - Dans le projet, trouvez `Samples` -> `PixelStreaming` -> `WebServers` et double cliquez sur `get_ps_servers.bat`
   - Ceci vous créera divers fichiers utiles au PixelStreaming.
   - Dans le dossier courant, trouvez `SignallingWebServer`-> `platform_scripts`-> `cmd`, puis double cliquez sur `setup.bat`.
   - Enfin, double cliquez sur `run_local.bat`.

3. **Accès au Stream**:

   - Ouvrez une nouvelle page dans votre navigateur et connectez-vous à votre localhost sur le port 80:

     `http://localhost:80`.

## **Démarrage quotidien du Stream** :

1. **Utilisation du Script** :

   - Un script à été créé afin de simplifier le lancement du pixel streaming, vous le trouverez dans le dossier `PINGOrange` et est nommé `lancementPixelStreaming`.
   - Si celui-c n'est pas présent, vous le trouverez dans le repo git fournit avec le livrable. Placez-le dans le dossier `PINGOrange`.
   - Enfin executez le fichier.

2. **Lancement Manuel**:

   - Lancez le raccourcis application créé lors du précédent point `"Premier démarrage du Stream/Raccourcis application"`.
   - Dans le projet, trouvez `Samples` -> `PixelStreaming` -> `WebServers` ->  `SignallingWebServer`-> `platform_scripts`-> `cmd`, puis double cliquez sur `setup.bat`.
   - Enfin, double cliquez sur `run_local.bat`.

   - Ouvrez une nouvelle page dans votre navigateur et connectez-vous à votre localhost sur le port 80:

     `http://localhost:80`.

     

## Dépannage et Ressources Supplémentaires

- Consultez la [documentation officielle](https://dev.epicgames.com/documentation/en-us/unreal-engine/getting-started-with-pixel-streaming-in-unreal-engine) pour des instructions détaillées et des solutions de dépannage.

---

Ce document est une introduction complète au Pixel Streaming dans Unreal Engine. Pour des informations supplémentaires et des guides avancés, veuillez consulter la documentation officielle d'Unreal Engine.
