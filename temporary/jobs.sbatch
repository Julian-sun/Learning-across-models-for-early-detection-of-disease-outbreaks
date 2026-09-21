#!/bin/bash

#SBATCH --job-name="aesop-detection-models (EVI)"

#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --output=detection_models.log
#SBATCH --error=detection_models.err

echo "---------------------------------------"
echo "Begin job: " $(date)
echo "---------------------------------------"

#singularity run /opt/images/cidacs/cidacs-jupyter-datascience-v1-r2.sif python sint_analise_test.py
singularity run /opt/images/cidacs/cidacs-jupyter-datascience-v1-r2.sif python sinte_int_per.py

echo "---------------------------------------"
echo "End job: " $(date)
echo "---------------------------------------"


