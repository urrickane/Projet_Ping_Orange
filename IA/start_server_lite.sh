#!/bin/bash

# Commande pour télécharger l'image Docker depuis le registre
docker pull iajibril.azurecr.io/iaorange:latest

# Affichage d'un message
echo "Image récupérée avec succès."
echo "Serveur lancé. Appuyer sur Ctrl+C pour le stopper."

# Commande pour exécuter l'image Docker téléchargée
docker run iajibril.azurecr.io/iaorange:latest
