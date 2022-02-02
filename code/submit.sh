#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --time=1:00:00
export WORKDIR=/beegfs/bashir/standalone/software
export HOME=/common/home/bashir
module purge

#module 2021a-norpath
#source ~/.bashrc
#. ~/anaconda3/etc/profile.d/conda.sh
#module spider Anaconda3
#eval "$(conda shell.bash hook)"
#source ${HOME}/.bashrc

#source ~/anaconda3/bin/activate
#conda activate ${HOME}/anaconda3/envs/jupyter_env
python ${WORKDIR}/code/daskdataframe.py
#${HOME}/anaconda3/envs/jupyter_env/bin/python   ${WORKDIR}/code/daskdataframe.py