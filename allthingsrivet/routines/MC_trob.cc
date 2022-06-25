#include "Rivet/Analysis.hh"
#include "Rivet/Projections/FinalState.hh"
#include "Rivet/Projections/ChargedFinalState.hh"
#include <iostream>
#include <fstream>
#include <chrono>
#include <cmath>
#include <vector>
#include <algorithm>



namespace Rivet {


  class MC_trob : public Analysis {
  public:

    /// Constructor
    MC_trob()
      : Analysis("MC_trob")
    {    }




    void init() {


        const ChargedFinalState cfs(Cuts::abseta < 2.5 && Cuts::pT > 500*MeV);
         declare(cfs, "CFS");
        // myfile << particle_entries << '\n';
        char particle_legend[400];

      sprintf( particle_legend,"     %9s ,%8s ,%4s",
               "particle_multiplicity","transverse_momenta_sum","beam_thrust");
      std::cout<<particle_legend<<std::endl;



    }


    /// Perform the per-event analysis
    void analyze(const Event& event) {



      const GenEvent* evt = event.genEvent();
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



     std::cout<<particle_entries<<std::endl;




      for (HepMC::GenEvent::particle_const_iterator p = evt->particles_begin(); p != evt->particles_end(); ++p) {

       }




    }

    void finalize() {}


  private:

    map<long, string> _pnames;


  };



  // The hook for the plugin system
  DECLARE_RIVET_PLUGIN(MC_trob);

}

