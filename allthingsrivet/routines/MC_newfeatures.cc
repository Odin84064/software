/*
// -*- C++ -*-
#include "Rivet/Analysis.hh"
#include "Rivet/Projections/FinalState.hh"
#include <iostream>
#include <fstream>
#include <chrono>


namespace Rivet {


  /// @author Dominic Hirschbuehl
  class MC_newfeatures : public Analysis {
  public:

    /// Constructor
    MC_newfeatures()
      : Analysis("MC_newfeatures")
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
        int p_stat = (*p)->status();

        double eta = (*p)->momentum().eta();

        double p_px = (*p)->momentum().px();
        double p_py = (*p)->momentum().py();
        double p_t = sqrt(pow(p_px,2) + pow(p_py,2));


        // Mass (prefer generated mass if available)




        if(p_stat == 1){
         char particle_entries[120];
         sprintf(particle_entries, " %+9.3g ,%9.3g",
              p_t,eta);

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
  DECLARE_RIVET_PLUGIN(MC_newfeatures);

}
*/

//
// Created by Bashir on 3/17/2022.
//

// -*- C++ -*-
#include "Rivet/Analysis.hh"
#include "Rivet/Projections/FinalState.hh"
#include <iostream>
#include <fstream>
#include <chrono>
#include <cmath>
#include <vector>
#include <algorithm>


namespace Rivet {



    class MC_newfeatures : public Analysis {
    public:

        void remove(std::vector<int> &v) {
            auto end = v.end();
            for (auto it = v.begin(); it != end; ++it) {
                end = std::remove(it + 1, end, *it);
            }

            v.erase(end, v.end());
            /*for (auto it = v.cbegin(); it != v.cend(); ++it) {
                std::cout << *it << ' ';
            }*/
        }


        void print_2d_vector(std::vector<std::vector<float> > &matrix) {
            for (unsigned int i = 0; i < matrix.size(); ++i) {
                for (unsigned int j = 0; j < matrix[i].size(); ++j) {
                    cout << matrix[i][j] << ", ";
                }
                cout << endl;
            }
            cout << endl;
        }

        std::vector<std::vector<float>>
        concatenate_pdg_id(std::vector<std::vector<float>> &matrix, std::vector<int> &v) {
            std::vector<std::vector<float> > matrix1;
            std::vector<float> matrix2;

            for (unsigned int k = 0; k < v.size(); ++k) {
                int counter = v[k];
                float total_num = 0;

                std::vector<float> sub{0, 0, 0, 0};
                sub[0] = counter;
                //sub[1] = 1;
                for (unsigned int i = 0; i < matrix.size(); ++i) {
                    if (matrix[i][0] == counter) {

                        total_num += 1;

                        for (unsigned int j = 2; j < sub.size(); ++j) {

                            sub[j] = sub[j] + matrix[i][j];

                        }
                    }

                }

                //std::for_each(sub.begin(), sub.end(), [mean](float &c){ c /= mean; });
                for (unsigned int l = 2; l < sub.size(); ++l)
                    sub[l] = sub[l] / total_num;

                sub[1] = total_num;
                //sub.insert(sub.begin(), total_num);
                matrix1.push_back(sub);
                sub.clear();

            }

            return matrix1;
        }

        template<typename T>
        std::vector<T> flatten(const std::vector<std::vector<T>> &v) {
            std::size_t total_size = 0;
            for (const auto &sub: v)
                total_size += sub.size(); // I wish there was a transform_accumulate
            std::vector<T> result;
            result.reserve(total_size);
            for (const auto &sub: v)
                result.insert(result.end(), sub.begin(), sub.end());
                cout<<endl;
            return result;
        }

        void print_vector(std::vector<float> &vector) {
            for (unsigned int i = 0; i < vector.size(); ++i) {
                cout << vector[i] << ", ";
            }
            cout << endl;
        }

        /// Constructor
        MC_newfeatures()
                : Analysis("MC_newfeatures") {}


        void init() {

        }


        /// Perform the per-event analysis
        void analyze(const Event &event) {

            //2d vector of all particles
            std::vector<std::vector<float> > arr;
            //2d vector of all particles

            //array of all PDG_ID
            std::vector<int> counter;
            vector<int>::iterator ip;
            // vector<int>::iterator it;




            const GenEvent *evt = event.genEvent();


            char particle_legend[120];

            sprintf(particle_legend, "    %8s ,%4s , %9s,%9s;",
                    "Total Number", "PDG_ID", "Pt", "eta ");


            std::cout << particle_legend << std::endl;
            for (HepMC::GenEvent::particle_const_iterator p = evt->particles_begin(); p != evt->particles_end(); ++p) {
                //std::cout<<"This is event :"<<p<< std::endl;

                int p_pdg_id = std::abs((*p)->pdg_id());
				double eta = (*p)->momentum().eta();

				double p_px = (*p)->momentum().px();
				double p_py = (*p)->momentum().py();
				double p_t = sqrt(pow(p_px,2) + pow(p_py,2));
				int p_stat = (*p)->status();










                if (p_stat == 1) {
                    //filling in particles with status 1 as a row of 2d vector array
                    //when no exits



                    std::vector<float> sub;
                    sub.push_back(p_pdg_id);
                    sub.push_back(p_stat);
                    sub.push_back(p_t);
                    sub.push_back(eta);
                    arr.push_back(sub);
                    sub.clear();
                }

            //char particle_entries[120];
             //sprintf(particle_entries, " %8i, %4i, %+9.3g ,%+9.3g ,%+9.3g ,%+9.3g",
                //p_pdg_id, p_stat, p_px, p_py, p_pz, p_pe);

                //std::cout<<particle_entries<<std::endl;



            }
            //filling the PDG ids in counter vector
            for (unsigned int i = 0; i < arr.size(); ++i) {

                counter.push_back(arr[i][0]);
            }

            remove(counter);


            std::cout << "\n" << std::endl;


            std::vector<std::vector<float> > arr1;

            arr1 = concatenate_pdg_id(arr, counter);
            //print_2d_vector(arr1);

            std::vector<float> arr2;
            arr2 = flatten(arr1);
            print_vector(arr2);


        }


        void finalize() {}


    private:

        map<long, string> _pnames;

    };

    DECLARE_RIVET_PLUGIN(MC_newfeatures);
}
