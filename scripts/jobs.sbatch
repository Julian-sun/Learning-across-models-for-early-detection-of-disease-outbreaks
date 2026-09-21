#!/bin/bash

#SBATCH --job-name="aesop-detection-models (EVI)"

#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=detection_models.log
#SBATCH --error=detection_models.err

echo "---------------------------------------"
echo "Begin job: " $(date)
echo "---------------------------------------"

# singularity run /opt/images/cidacs/cidacs-jupyter-datascience-v1-r2.sif python -m pip install -r ../requirements.txt

# singularity run /opt/images/cidacs/cidacs-jupyter-datascience-v1-r2.sif python def_sea_peri_MEM_all_cities.py
# singularity run /opt/images/cidacs/cidacs-jupyter-datascience-v1-r2.sif python baseline_epi_thresh_mem_all_city.py
# singularity run /opt/images/cidacs/cidacs-jupyter-datascience-v1-r2.sif python surge_aps_with_MEM.py
singularity run /opt/images/cidacs/cidacs-jupyter-datascience-v1-r2.sif python temporary/EVI_selec_para.py

echo "---------------------------------------"
echo "End job: " $(date)
echo "---------------------------------------"
