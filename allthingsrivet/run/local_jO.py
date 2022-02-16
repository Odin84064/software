#https://gitlab.cern.ch/atlas-physics/pmg/tutorials/-/blob/master/Rivet/local_jO.py
from AthenaCommon.AppMgr import theApp
theApp.EvtMax = 10000

import AthenaPoolCnvSvc.ReadAthenaPool
import glob
#svcMgr.EventSelector.InputCollections = glob.glob("/lustre/hirsch/sfsscratch/PoolFiles/" + str(dataset) + "/*")
#svcMgr.EventSelector.InputCollections = glob.glob("/lustre/shayma/Derivations/evgen/" + str(dataset) + "/*")

#svcMgr.EventSelector.InputCollections = glob.glob("/beegfs/shayma/RivetVali/" + str(dataset) )
#svcMgr.EventSelector.InputCollections = glob.glob("/beegfs/shayma/Derivations/" + str(dataset) + "/*")
svcMgr.EventSelector.InputCollections = glob.glob("/beegfs/hirsch/sfsscratch/PoolFiles/" + str(dataset) + "/*")
MessageSvc.defaultLimit = 9999999 
print(svcMgr.EventSelector.InputCollections)
from AthenaCommon.AlgSequence import AlgSequence
job = AlgSequence()

from Rivet_i.Rivet_iConf import Rivet_i
rivet = Rivet_i()
import os
#rivet.AnalysisPath ="/beegfs/shayma/Rivet/routines/"
rivet.AnalysisPath ="/beegfs/bashir/standalone/alllthingsrivet/routines/"

rivet.Analyses += [ 'MC_ttbar_ColorReconnection' ]
rivet.RunName = ''
rivet.HistoFile = str(dataset) + ""
rivet.HistoFile += '_MyOutput.yoda.gz'
rivet.CrossSection = 1.0
#rivet.IgnoreBeamCheck = True
rivet.AddMissingBeamParticles = True #to run over DAOD
job += rivet

