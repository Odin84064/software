#!/bin/bash
#SBATCH --job-name=Electron211_6layers
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32GB
export WORKDIR=/beegfs/bashir/standalone
source ${WORKDIR}/virtual_env/env/bin/activate
python ${WORKDIR}/software/code/Validation_Photon.py 2>&1 | tee   ${WORKDIR}/software/code/Electron211_6layers.txt

