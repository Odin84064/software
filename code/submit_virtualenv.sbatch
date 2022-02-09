#!/bin/bash
#SBATCH --job-name=testdata
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
export WORKDIR=/beegfs/bashir/standalone
source ${WORKDIR}/virtual_env/env/bin/activate
python ${WORKDIR}/software/code/EDAonCluster.py

