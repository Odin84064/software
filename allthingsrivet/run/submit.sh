#! /bin/bash

export JO=$1
export DATASET=$2
export WORKDIR=/beegfs/bashir/standalone/software/allthingsrivet

if [ -z $3 ]; then
  export RUN=`echo $DATASET | awk -F. '{print $2}'`
else
  export RUN=`echo $DATASET | awk -F. '{print $1}'`
fi

#mkdir /beegfs/shayma/Rivet/logs/
#sbatch -o ${WORKDIR}/logs/submit_${RUN}.out -e ${WORKDIR}/logs/submit_${RUN}.err -p normal --job-name Rivet_${RUN} --export WORKDIR,RUN,JO,DATASET run_rivet_onnode.sh
sbatch -o ${WORKDIR}/logs/submit_${RUN}.out -e ${WORKDIR}/logs/submit_${RUN}.err  -p normal --job-name   Rivet_${RUN} --export WORKDIR,RUN,JO,DATASET run_rivet_onnode.sh

