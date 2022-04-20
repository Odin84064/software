#! /bin/bash


cd $WORKDIR

export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
source ${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh
asetup 21.6.72, AthGeneration #Rivet version  3.1.4

# create temp directory
wn=`hostname | awk -F. '{print $1}'`
tmpdir=/tmp/${wn}_${RUN}_${NUM}
 
mkdir $tmpdir
cd $tmpdir

cp $WORKDIR/allthingsrivet/run/$JO .

export RIVET_REF_PATH=/beegfs/bashir/standalone/software/allthingsrivet/routines
export RIVET_ANALYSIS_PATH=/beegfs/bashir/standalone/software/allthingsrivet/routines

cat $JO

#athena $JO -c "dataset=\"${DATASET}\""   2>&1 | tee  /beegfs/bashir/standalone/yoda/output.txt

athena ${WORKDIR}/jO/runRivet.py -c "dataset=\"mc15_valid.950214.PhPy8_ttbar_CMCpwg_Monash_valid.evgen.EVNT.e8324_tid24450207_00\"" 2>&1 | tee  /beegfs/bashir/standalone/software/allthingsrivet/logs/mc15validconsolidated.txt
athena  ${WORKDIR}/jO/runRivet.py -c "dataset=\"mc15_13TeV.950507.PhH7EG_ttbar_hdamp258p75_nonallhad_cluster_valid.evgen.EVNT.e8419\"" 2>&1 | tee /beegfs/bashir/standalone/software/allthingsrivet/logs/mc1513tevconsolidated.txt

ls 

#cp *.yoda.gz /beegfs/shayma/Rivet/yoda/
#cp *.yoda.gz /beegfs/bashir/standalone/allthingsrivet/yoda/
#cp *.txt    /beegfs/bashir/standalone/yoda/
#
#cp example.txt  /beegfs/bashir/standalone/yoda/
# clean up
cd $WORKDIR
rm -rf $tmpdir
