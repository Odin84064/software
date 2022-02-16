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


   // std:: ofstream MyFile("filename.txt");

    /// @name Analysis methods
    //@{

    void init() {
      //FinalState fs;
     // const FinalState fs(Cuts::abseta < 5);
    // const Event event;
     //const Event& event;
   //  std:: ofstream MyFile("filename.txt");
    }


    /// Perform the per-event analysis
    void analyze(const Event& event) {
      
      const GenEvent* evt = event.genEvent();

      //cout << string(120, '=') << "\n" << endl;

      // Weights
     // cout << "Weights(" << evt->weights().size() << ")=";
      /// @todo Re-enable
      // for (double w,  evt->weights())
      //   cout << w << " ";
      
      std:: ofstream MyFile("filename.txt");
      
      // Print a legend to describe the particle info
      char particle_legend[120];
      //sprintf( particle_legend,"     %9s %8s %4s (%9s,%9s,%9s,%9s,%9s)",
        //       "Barcode","PDG ID","Status","Px","Py","Pz","E ","m");
      sprintf( particle_legend,"     %9s ,%8s ,%4s , %9s,%9s ,%9s ,%9s ,%9s;",
               "Barcode","PDG ID","Status","Px","Py","Pz","E ","m");
     // auto begin = std::chrono::high_resolution_clock::now(); 
      MyFile << particle_legend << '\n';
      //cout << endl;
     // cout << "                                       GenParticle Legend\n" << particle_legend << "\n";
      //std:: ofstream MyFile("filename.txt");
      // Print all particles
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


       // char particle_entries[120];
       // sprintf(particle_entries, " %9i %8i %4i   (%+9.3g,%+9.3g,%+9.3g,%+9.3g,%9.3g)",
        //        p_bcode, p_pdg_id, p_stat, p_px, p_py, p_pz, p_pe, p_mass);
       
        if(p_stat == 1){
         char particle_entries[120];
         sprintf(particle_entries, " %9i,  %8i, %4i, %+9.3g ,%+9.3g ,%+9.3g ,%+9.3g ,%9.3g;",
                p_bcode,p_pdg_id, p_stat, p_px, p_py, p_pz, p_pe, p_mass);
         cout << particle_entries << "\n";
         MyFile << particle_entries << '\n'; 
         }
        


      }
     // auto end = std::chrono::high_resolution_clock::now();
     // auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
      cout << "\n" << endl;
      MyFile.close();
     // printf("Time measured: %.3f seconds.\n", elapsed.count() * 1e-9);
    }


    /// Normalise histograms etc., after the run
    void finalize() {}
     // auto end = std::chrono::high_resolution_clock::now();
      //auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
      //cout << "\n" << endl;
     // MyFile.close();
    // printf("Time measured: %.3f seconds.\n", elapsed.count() * 1e-9);    }

    //@}

  private:

    map<long, string> _pnames;


  };



  // The hook for the plugin system
  DECLARE_RIVET_PLUGIN(MC_DUMP);

}
