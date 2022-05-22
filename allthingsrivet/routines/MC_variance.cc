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

        void remove(std::vector<int> &v) {
            auto end = v.end();
            for (auto it = v.begin(); it != end; ++it) {
                end = std::remove(it + 1, end, *it);
            }

            v.erase(end, v.end());

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

                std::vector<float> sub{0, 0, 0, 0,0,0};

                sub[0] = counter;

                for (unsigned int i = 0; i < matrix.size(); ++i) {
                    if (matrix[i][0] == counter) {

                        total_num += 1;

                        for (unsigned int j = 2; j < sub.size()-2; ++j) {

                            sub[j] = sub[j] + matrix[i][j];

                        }
                    }

                }


                for (unsigned int l = 2; l < sub.size()-2; ++l)
                    sub[l] = sub[l] / total_num;


               for (unsigned int i = 0; i < matrix.size(); ++i) {
                    if (matrix[i][0] == counter) {



                        for (unsigned int j = 2; j < sub.size()-2; ++j) {

                            sub[j+2] = sub[j+2] + (matrix[i][j]- sub[j]) * (matrix[i][j] - sub[j]);


                        }
                    }

                }

                sub[1] = total_num;
                sub[4] /= total_num;
                sub[5] /= total_num;
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
        MC_variance()
                : Analysis("MC_variance") {}


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

            const GenEvent *evt = event.genEvent();


            char particle_legend[120];

            sprintf(particle_legend, " %8s ,%4s, %9s,%9s, %9s,%9s;",
                    "Total Number", "PDG_ID", "Pt", "eta ","Var_Pt","Var_eta");


            std::cout << particle_legend << std::endl;
            for (HepMC::GenEvent::particle_const_iterator p = evt->particles_begin(); p != evt->particles_end(); ++p) {

                double p_pdg_id = std::abs((*p)->pdg_id());
				double eta = std::abs((*p)->momentum().eta());

				double p_px = (*p)->momentum().px();
				double p_py = (*p)->momentum().py();
				double p_t = sqrt(pow(p_px,2) + pow(p_py,2));
				double var_p_t =0;
				double var_eta = 0;

				double p_stat = (*p)->status();
				//int n_11 =0;
				//int n_13 = 0;

                 if (p_stat == 1) {
                   // if(p_pdg_id ==11)
                   // {n_11 +=1;}
                   // if(p_pdf_id ==13)
                   // {n_13 +=1;}
                    std::vector<float> sub;
                    sub.push_back(p_pdg_id);
                    sub.push_back(p_stat);
                    sub.push_back(p_t);
                    sub.push_back(eta);
                    sub.push_back(var_p_t);
                    sub.push_back(var_eta);


                    arr.push_back(sub);
                    sub.clear();
                }
                //std::vector<float> n_11_13;
               // n_13_11.push_back(n_11);
                //n_13_11.push_back(n_13);
                //arr.push_back(n_13_11);
                //n_13_11.clear()



            }
            //filling the PDG ids in counter vector
            for (unsigned int i = 0; i < arr.size(); ++i) {

                counter.push_back(arr[i][0]);
            }

            remove(counter);


            std::cout << "\n" << std::endl;


            std::vector<std::vector<float> > arr1;

            arr1 = concatenate_pdg_id(arr, counter);
            std::vector<float> arr2;
            arr2 = flatten(arr1);
            print_vector(arr2);


        }


        void finalize() {}


    private:

        map<long, string> _pnames;

    };

    DECLARE_RIVET_PLUGIN(MC_variance);
}
