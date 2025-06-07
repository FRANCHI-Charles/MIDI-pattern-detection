#!/bin/bash

#SBATCH --partition=gpu    # Partition de calcul par défaut
#SBATCH --gres=gpu:1    # Nombre de GPU
#SBATCH --job-name=traincnn   # Nom de la tâche
#SBATCH --time=48:00:00        # Durée maximale de la tâche (HH:MM:SS)
#SBATCH --nodes=1             # Nombre de noeuds
#SBATCH --ntasks-per-node=32  # Nombre de tâches par noeud
#SBATCH --ntasks-per-core 1   # Nombre de tâches par coeur (1 pour éviter l'hyperthreading)

# Optionnel : Envoie un email à la fin du job
#SBATCH --mail-type=END       # Type d'email (BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=c.franchi@unistra.fr

export MIOPEN_USER_DB_PATH="/tmp/my-miopen-cache"
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
rm -rf ${MIOPEN_USER_DB_PATH}
mkdir -p ${MIOPEN_USER_DB_PATH}
python3 cnn_training.py
