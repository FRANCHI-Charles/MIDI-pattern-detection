#!/bin/bash

#SBATCH --partition=gpu    # Partition de calcul par défaut
#SBATCH --gres=gpu:1    # Nombre de GPU
#SBATCH --job-name=traincnn   # Nom de la tâche
#SBATCH --time=24:00:00        # Durée maximale de la tâche (HH:MM:SS)
#SBATCH --nodes=4             # Nombre de noeuds
#SBATCH --ntasks-per-node=32  # Nombre de tâches par noeud
#SBATCH --ntasks-per-core 1   # Nombre de tâches par coeur (1 pour éviter l'hyperthreading)

# Optionnel : Envoie un email à la fin du job
#SBATCH --mail-type=END       # Type d'email (BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=c.franchi@math.unistra.fr

srun rocm-smi
python3 cnn_training.py
