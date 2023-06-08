#include"../Timer.cpp"
#include<vector>
#include<tuple>
#include<omp.h>
#include<iostream>
#include <cstdlib> 
#include <ctime>
#include "conjugate_gradient.cpp"
#include "cluster.cpp"
using namespace std;

using T = double;

int number_of_simulations = 1000;   //The total number of simulations run per cluster size. The lowest result will be output.
int smallest_cluster = 2;           //The number of atoms in the smallest cluster computed.
int largest_cluster = 20;           //The number of atoms in the largest cluster computed.
bool log_results = false;           //Wether or not the best structures will be saved in the Results folder.


int main() {
    using Vector = Eigen::Matrix<T, -1, 1>;

    srand(time(NULL));
    Timer a;
    for (int i = smallest_cluster; i<=largest_cluster; i++) {
        cluster_optimizer<T> c(i,number_of_simulations,pow(10,-14));
        auto x_best = c.optimize(log_results);

        std::cout << i << ": " << std::endl;
        std::cout << "Best Energy: " << std::get<1>(x_best) << std::endl; 
        std::cout << "Best Energy (Leary): " << std::get<2>(x_best) << std::endl << std::endl; 
    }
}