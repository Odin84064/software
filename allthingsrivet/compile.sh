#rm routines/RivetMCDump.so
#rivet-build routines/RivetMCDump.so routines/MC_DUMP.cc -Wno-deprecated-declarations
#rm routines/RivetMCDumpevent.so
#rivet-build routines/RivetMCDumpevent.so routines/MC_DUMP_event.cc -Wno-deprecated-declarations
rm routines/RivetMCnewfeatures.so
rivet-build routines/RivetMCnewfeatures.so routines/MC_newfeatures.cc -Wno-deprecated-declarations
#rm routines/RivetnewMCDump.so
#rivet-build routines/RivetnewMCDump.so routines/newMC_DUMP.cc -Wno-deprecated-declarations
#rm routines/RivetLP.so
#rivet-build routines/Rivetlp.so routines/lp.cc -Wno-deprecated-declarations
