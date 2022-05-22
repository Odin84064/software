#include "Rivet/Analysis.hh"
#include "Rivet/Projections/FinalState.hh"
#include <iostream>
#include <fstream>
#include <chrono>
#include <cmath>
#include <vector>
#include <algorithm>


namespace Rivet {



    class MC_variance : public Analysis {
    public:



        void print_2d_vector(std::vector<std::vector<float> > &matrix) {
            for (unsigned int i = 0; i < matrix.size(); ++i) {
                for (unsigned int j = 0; j < matrix[i].size(); ++j) {
                    cout << matrix[i][j] << ", ";
                }
                cout << endl;
            }
            cout << endl;
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
        MC_variance()
                : Analysis("MC_1113") {}


        void init() {

        }


        /// Perform the per-event analysis
        void analyze(const Event &event) {

            //2d vector of all particles
            std::vector<std::vector<float> > arr;
            //2d vector of all particles
            const GenEvent *evt = event.genEvent();
            int n_11 =0;
		    int n_13 = 0;
            for (HepMC::GenEvent::particle_const_iterator p = evt->particles_begin(); p != evt->particles_end(); ++p) {

                double p_pdg_id = std::abs((*p)->pdg_id());
                double p_stat = (*p)->status();




                 if (p_stat == 1) {
                   if(p_pdg_id ==11)
                   {n_11 +=1;}
                   if(p_pdg_id ==13)
                    {n_13 +=1;}



                }




            }
            std::vector<float> n_11_13;
                n_11_13.push_back(n_11);
                n_11_13.push_back(n_13);
                arr.push_back(n_11_13);
                n_11_13.clear();


            std::cout << "\n" << std::endl;


            //std::vector<std::vector<float> > arr1;

            //arr1 = concatenate_pdg_id(arr, counter);
            //std::vector<float> arr2;
           // arr2 = flatten(arr1);
            print_2d_vector(arr);


        }


        void finalize() {}


    private:

        map<long, string> _pnames;

    };

    DECLARE_RIVET_PLUGIN(MC_variance);
}
