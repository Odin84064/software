## basic script to run any python job on cluster with the virtual environment 'env'gi
##identify std output and std err location if any
## change the python executable in last line



#!/bin/bash
#SBATCH --job-name=testdata
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32GB

export WORKDIR=/beegfs/bashir/standalone
source ${WORKDIR}/virtual_env/env/bin/activate
python ${WORKDIR}/software/code/daskdataframe.py