#include "Rivet/Analysis.hh"
#include "Rivet/Projections/FinalState.hh"
#include "Rivet/Projections/ChargedFinalState.hh"
#include "Rivet/Analyses/MC_JetAnalysis.hh"

#include "Rivet/Projections/FastJets.hh"

#include <iostream>
#include <fstream>
#include <chrono>
#include <cmath>
#include <vector>
#include <algorithm>



namespace Rivet {


  class MC_jets : public MC_JetAnalysis {
  public:

    /// Constructor
    MC_jets()
      :  MC_JetAnalysis("MC_jets", 4, "Jets",5*GeV)
    {    }





    void init() {

      FinalState fs;
      FastJets jetpro(fs, FastJets::ANTIKT, 0.4);
      declare(jetpro, "Jets");
      char particle_legend[400];

      sprintf( particle_legend,"     %9s ,%9s ,%9s, %9s",
               "5GeV","10GeV","15GeV","20GeV");
      std::cout<<particle_legend<<std::endl;



       /* const ChargedFinalState cfs(Cuts::abseta < 2.5 && Cuts::pT > 500*MeV);
         declare(cfs, "CFS");
        // myfile << particle_entries << '\n';
        char particle_legend[400];

      sprintf( particle_legend,"     %9s ,%8s ,%4s",
               "particle_multiplicity","transverse_momenta_sum","beam_thrust");
      std::cout<<particle_legend<<std::endl; */



    }

  void analyze(const Event & e){
    /// Perform the per-event analysis

    //MC_JetAnalysis::analyze(e);
    //double _jetptcut = 10 *GeV;
    const Jets& jets = apply<FastJets>(e, _jetpro_name).jetsByPt(5*GeV);
    const Jets& jets1 = apply<FastJets>(e, _jetpro_name).jetsByPt(10*GeV);
    const Jets& jets2 = apply<FastJets>(e, _jetpro_name).jetsByPt(15*GeV);
    const Jets& jets3 = apply<FastJets>(e, _jetpro_name).jetsByPt(20*GeV);


      /*const GenEvent* evt = event.genEvent();
      const ChargedFinalState& cfs = apply<ChargedFinalState>(event, "CFS");

      int cpm = cfs.size();

      double sumPt = 0.0, beamThrust = 0.0;
      for (const Particle& p : cfs.particles()) {
       const double pT = p.pT();
        sumPt += pT;
        beamThrust += pT*exp(-p.abseta());}

      char particle_entries[120];
      sprintf(particle_entries, " %9i, %+9.3g , %+9.3g",
              cpm,sumPt,beamThrust);



     std::cout<<particle_entries<<std::endl;*/
    float jet_size = jets.size();
    float jet_size1 = jets1.size();
    float jet_size2 = jets2.size();
    float jet_size3 = jets3.size();
    char particle_entries[200];
    sprintf(particle_entries, "%f," "%f," "%f," "%f",
              jet_size,jet_size1,jet_size2,jet_size3);
    std::cout<<particle_entries<<std::endl;








    }

    void finalize() {}


  private:

    map<long, string> _pnames;


  };



  // The hook for the plugin system
  DECLARE_RIVET_PLUGIN(MC_jets);

}

