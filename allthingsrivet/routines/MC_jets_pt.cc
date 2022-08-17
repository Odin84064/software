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


  class MC_jets_pt : public MC_JetAnalysis {
  public:

    /// Constructor
    MC_jets_pt()
      :  MC_JetAnalysis("MC_jets_pt", 4, "Jets",5*GeV)
    {    }





    void init() {

      FinalState fs;
      FastJets jetpro(fs, FastJets::ANTIKT, 0.4);
      declare(jetpro, "Jets");
      char particle_legend[800];

      //sprintf( particle_legend," %9s ,%9s ,%9s, %9s, %9s,%9s ,%9s ,%9s, %9s, %9s",
       //        "5GeV","10GeV","15GeV","20GeV","50GeV");
       sprintf( particle_legend,"%9s,%9s,%9s,%9s,%9s,%9s,%9s,%9s,%9s,%9s","jetsize_5","sumpt_5","jetsize_10","sumpt_10","jetsize_15","sumpt_15","jetsize_20","sumpt_20","jetsize_50","sumpt_50");
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
    const Jets& jets5 = apply<FastJets>(e, _jetpro_name).jetsByPt(5*GeV);
    const Jets& jets10 = apply<FastJets>(e, _jetpro_name).jetsByPt(10*GeV);
    const Jets& jets15 = apply<FastJets>(e, _jetpro_name).jetsByPt(15*GeV);
    const Jets& jets20 = apply<FastJets>(e, _jetpro_name).jetsByPt(20*GeV);
    const Jets& jets50 = apply<FastJets>(e, _jetpro_name).jetsByPt(50*GeV);


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
    float jet_size5 = jets5.size();
    float jet_size10 = jets10.size();
    float jet_size15 = jets15.size();
    float jet_size20 = jets20.size();
    float jet_size50 = jets50.size();
    //char particle_entries[400];
    //sprintf(particle_entries, "%f," "%f," "%f," "%f","%f",
    //          jet_size,jet_size1,jet_size2,jet_size3,jet_size4);



    double HT5=0.0;
    double HT10=0.0;
    double HT15=0.0;
    double HT20=0.0;
    double HT50=0.0;
    /*foreach (const Jet& jet, jets) {
    HT += jet.pT();
    }*/
    for(int i =0;i < jet_size5;i++){
        HT5 += jets5[i].pt();

    }
    for(int i =0;i < jet_size10;i++){
        HT10 += jets10[i].pt();

    }
    for(int i =0;i < jet_size15;i++){
        HT15 += jets15[i].pt();

    }
    for(int i =0;i < jet_size20;i++){
        HT20 += jets20[i].pt();

    }
    for(int i =0;i < jet_size50;i++){
        HT50 += jets50[i].pt();

    }

std::cout<<jet_size5<<","<<HT5<<","<<jet_size10<<","<<HT10<<","<<jet_size15<<","<<HT15<<","<<jet_size20<<","<<HT20<<","<<jet_size50<<","<<HT50<<std::endl;


    }

    void finalize() {}


  private:

    map<long, string> _pnames;


  };



  // The hook for the plugin system
  DECLARE_RIVET_PLUGIN(MC_jets_pt);

}

