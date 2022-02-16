// -*- C++ -*-
#include "Rivet/Analysis.hh"
#include "Rivet/Projections/FinalState.hh"
#include <iostream>
#include <fstream>
#include <chrono>


namespace Rivet {


  /// @author Dominic Hirschbuehl
  class MC_DUMP : public Analysis {
  public:

    /// Constructor
    MC_DUMP()
      : Analysis("MC_DUMP")
    {    }


   

    void init() {
     
    }


    /// Perform the per-event analysis
    void analyze(const Event& event) {
      
      const GenEvent* evt = event.genEvent();

      
      
      

      
      //std::ofstream myfile;
       //myfile.open ("example.txt");
  
      
      
      char particle_legend[120];
      
      sprintf( particle_legend,"     %9s ,%8s ,%4s , %9s,%9s ,%9s ,%9s ,%9s;",
               "Barcode","PDG ID","Status","Px","Py","Pz","E ","m");
      
      // myfile<< particle_legend << '\n';
      //	std::cout.flush();
       std::cout<<particle_legend<<std::endl;     
      for (HepMC::GenEvent::particle_const_iterator p = evt->particles_begin(); p != evt->particles_end(); ++p) {
        int p_bcode = (*p)->barcode();
        int p_pdg_id = (*p)->pdg_id();
        double p_px = (*p)->momentum().px();
        double p_py = (*p)->momentum().py();
        double p_pz = (*p)->momentum().pz();
        double p_pe = (*p)->momentum().e();

        int p_stat = (*p)->status();
        
        // Mass (prefer generated mass if available)
        double p_mass = (*p)->generated_mass();
        if (p_mass == 0 && !(p_stat == 1 && p_pdg_id == 22)) p_mass = (*p)->momentum().m();


     
       
        if(p_stat == 1){
         char particle_entries[120];
         sprintf(particle_entries, " %9i,  %8i, %4i, %+9.3g ,%+9.3g ,%+9.3g ,%+9.3g ,%9.3g;",
              p_bcode,p_pdg_id, p_stat, p_px, p_py, p_pz, p_pe, p_mass);
         
        // myfile << particle_entries << '\n'; 
         std::cout<<particle_entries<<std::endl;         }
        


      }
     
     std::cout << "\n" << std::endl;
     std::cout.flush();
     // myfile.close();
     
    }

    void finalize() {}
     

  private:

    map<long, string> _pnames;


  };



  // The hook for the plugin system
  DECLARE_RIVET_PLUGIN(MC_DUMP);

}
