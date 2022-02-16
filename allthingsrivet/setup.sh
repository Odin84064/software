source /cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase/user/atlasLocalSetup.sh
asetup 21.6.69,AthGeneration
export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
alias setupATLAS='source ${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh'
source setupRivet.sh

export RIVET_REF_PATH=$PWD/routines
export RIVET_ANALYSIS_PATH=$PWD/routines

