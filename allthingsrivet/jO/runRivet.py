## job option script to run Rivet inside Athena

#from AthenaCommon.AppMgr import ServiceMgr as svcMgr

import glob
import AthenaPoolCnvSvc.ReadAthenaPool
svcMgr.EventSelector.InputCollections = glob.glob("/beegfs/hirsch/sfsscratch/PoolFiles/" + str(dataset) + "/*")

# limit number of events
from AthenaCommon.AppMgr import theApp
theApp.EvtMax = 1000
MessageSvc.defaultLimit = 9999999999999
## Now set up Rivet
from Rivet_i.Rivet_iConf import Rivet_i
rivet=Rivet_i("Rivet")
#rivet.AnalysisPath ="/lustre/hirsch/Athena/Rivet/v3.1.2/routines/"
rivet.Analyses += ["MC_DUMP"]
rivet.CrossSection = 1

rivet.HistoFile = "dummy.yoda"
rivet.SkipWeights=True

from AthenaCommon.AlgSequence import AlgSequence
topSequence = AlgSequence()
topSequence += rivet


