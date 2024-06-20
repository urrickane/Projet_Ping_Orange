

# Unreal Engine

## Table des matières

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Structure du Projet](#structure-du-projet)
4. [Utilisation](#utilisation)

## Introduction

Unreal Engine est un moteur de jeu puissant développé par Epic Games. Ce dépôt contient un projet Unreal Engine qui illustre un MetaHuman ayant pour but, l'accueil de la clientèle en boutique Orange.

## Installation

### Prérequis

- [Unreal Engine](https://www.unrealengine.com/download)
- [Git](https://git-scm.com/)
- [Visual Studio](https://visualstudio.microsoft.com/) (pour le développement sous Windows)

### Étapes d'installation

1. **Cloner le dépôt**

   ```bash
   git clone https://github.com/votre-utilisateur/votre-projet-unreal.git
   cd votre-projet-unreal
   ```

2. **Ouvrir le projet dans Unreal Engine**

   Lancez Unreal Engine, puis cliquez sur `Open Project` et sélectionnez le fichier `.uproject` situé dans le répertoire cloné.

3. **Compiler le projet**

   Si des modifications de code ont été effectuées, vous devrez peut-être compiler le projet en utilisant Visual Studio.

## Structure du Projet

Voici un aperçu de la structure du projet :

```
Content/
├── Camera/             		# Déclaration de la caméra
├── MetaHumans/          
│   └── Vivian/ 						# Dossier relatif au MetaHuman
├── Mixamo 
│   └── Bonjour/						# Dossier d'animations séquence Bonjour
│   └── idle/								# Dossier d'animations séquence Idle
│   └── AB_Idle							#	Animation BluePrint séquence Idle
├── Player/ 
│   └── BP_GameModeOrange		# BluePrint GameMode
│   └── BP_Input						# BluePrint Gestion du joueur
│   └── Mybuttonwidget			# Widget débogage, génération du click bonjour
└── OrangeMapFinal					# Level de la scène
```

## Utilisation

### Lancer le Projet

1. Ouvrez le projet dans Unreal Engine.
2. Cliquez sur `Play` pour démarrer le jeu ou la simulation.

### Développement

Pour développer des fonctionnalités supplémentaires, vous pouvez ajouter des blueprints, des assets, ou du code C++ dans les répertoires appropriés. N'oubliez pas de recompiler le projet après avoir ajouté ou modifié du code C++.

---

Ce document est une introduction au projet Orange PING de Télécom Saint-Étienne dans Unreal Engine. Pour des informations supplémentaires et des guides avancés, veuillez consulter la documentation officielle d'Unreal Engine.
