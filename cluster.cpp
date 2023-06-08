#pragma once
#include "conjugate_gradient.cpp"
#include<cmath>
#include<tuple>
#include <fstream>
#include<string>
#include<iomanip>

template<typename T>
class cluster_optimizer {
    using Vector = Eigen::Matrix<T, -1, 1>;

    struct distance {
        T squared;
        T dx;
        T dy;
        T dz;
    };

    int N_atoms;
    int N_runs;
    T tolerance;
    Vector best_x;
    T best_energy = pow(10,20);

    static T u(T r_squared) {
        constexpr T epsilon = 1.;
        constexpr T sigma = pow(1. , 2);
        constexpr T rc = pow(2.5*sigma, 2);

        constexpr T rc_sigma = - pow(rc/sigma, -6) + pow(rc/sigma, -3);

        if (r_squared <= rc && r_squared>0.01) {
            return 4 * epsilon * ( pow(r_squared/sigma, -6) - pow(r_squared/sigma, -3) + rc_sigma );
        }
        return 0;
    }

    static distance calculate_distance_squared(T x1, T y1, T z1, T x2, T y2, T z2, T L) {
        T dx = x2 - x1 - L * round( (x2 - x1)/L );
        T dy = y2 - y1 - L * round( (y2 - y1)/L );
        T dz = z2 - z1 - L * round( (z2 - z1)/L );
        return {pow(dx, 2) + pow(dy, 2) + pow(dz, 2), dx, dy, dz};
    }

    static T calculate_energy(const Vector& x, T L) {

        T alpha = 0.0001 * pow(x.rows() / 3, -2./3);
        T first_sum = 0;

        #pragma omp simd
        for(int i=0; i < x.rows(); i+=3) {
            first_sum += pow(x[i], 2) + pow(x[i+1], 2) + pow(x[i+2], 2);
        } 

        
        T second_sum = 0;
        for(int i=0; i < x.rows(); i+=3) {
            for(int j=i+3; j < x.rows(); j+=3) {
                T r_squared = calculate_distance_squared(x[i], x[i+1], x[i+2], x[j], x[j+1], x[j+2], L).squared;
                second_sum += u(r_squared);
            }
        }
        return alpha * first_sum + second_sum;
    };

    static T calculate_leary_energy(Vector x, T L) {
        constexpr T epsilon = 1.;
        constexpr T sigma = pow(1. , 2);

        T second_sum = 0;
        #pragma omp simd
        for(int i=0; i < x.rows(); i+=3) {
            for(int j=i+3; j < x.rows(); j+=3) {
                T r_squared = calculate_distance_squared(x[i], x[i+1], x[i+2], x[j], x[j+1], x[j+2], L).squared;
                second_sum += (4 * epsilon * ( pow(r_squared/sigma, -6) - pow(r_squared/sigma, -3) ));
            }
        }
        return second_sum;
    };

    static Vector calculate_forces(const Vector& x, T L) {
        constexpr T epsilon = 1.;
        constexpr T sigma = 1.;
        constexpr T rc = 2.5*sigma;
        const T center_factor = 2 * 0.0001*pow(x.rows() / 3, -2./3);

        Vector Forces = Vector(x.rows());

        #pragma omp simd
        for(int i=0; i < x.rows(); i+=3) {
            for(int j=i+3; j < x.rows(); j+=3) {
                auto R = calculate_distance_squared(x[i], x[i+1], x[i+2], x[j], x[j+1], x[j+2], L);
                T r = pow(R.squared, 0.5);

                if (r <= rc) {
                    T lennard_jones_factor = 4 * epsilon * (-12 * pow(r/sigma, -13)  + 6 * pow(r/sigma, -7))/sigma;

                    Forces[i] += R.dx/r * lennard_jones_factor;
                    Forces[j] -= R.dx/r * lennard_jones_factor;

                    Forces[i+1] += R.dy/r * lennard_jones_factor;
                    Forces[j+1] -= R.dy/r * lennard_jones_factor;

                    Forces[i+2] += R.dz/r * lennard_jones_factor;
                    Forces[j+2] -= R.dz/r * lennard_jones_factor;
                }
            }
        }

        #pragma omp simd
        for(int i=0; i < x.rows(); i+=3) {
            Forces[i] -=  center_factor * x[i];
            Forces[i+1] -=  center_factor * x[i+1];
            Forces[i+2] -=  center_factor * x[i+2];
        }

        return Forces;
    }

    static Vector keep_in_box(Vector x, T L) {
        #pragma omp simd
        for(int i = 0; i<x.rows(); i++) {
            if(abs(x[i]) > L/2) {
                x[i] -= ( L * round( x[i] /L ) );
            }
        }
        return x;
    }



public: 
    cluster_optimizer(int N_atoms, int N_runs, T tolerance) : N_atoms(N_atoms), N_runs(N_runs), tolerance(tolerance) {}

    std::tuple<Vector,T,T> optimize(bool logging=false) {
        constexpr T N_by_V = 0.01;

        T L = pow(N_atoms/N_by_V, 1./3);
        
        #pragma omp parallel for schedule(dynamic)
        for(int i= 0; i<N_runs; i++) {

            Vector x =  L/50 * Vector::Random(N_atoms * 3);

            conjugate_gradient_optimizer<T> opt(x, calculate_energy, calculate_forces, keep_in_box, tolerance, L);
            x = opt.optimize();

            T energy = calculate_energy(x, L);

            #pragma omp critical
            {
                if (energy<best_energy) {
                    best_energy = energy;
                    best_x = x;
                }
            }
        }

        if(logging) {
            std::ofstream file;
            file.open ((std::string)"Results/"+ std::to_string(N_atoms) + (std::string)".dat", std::fstream::out);
            if (file.is_open()) {
                file << "Lowest energy: " << std::setprecision(10) << best_energy << std::endl;
                file << "Leary method for energy: " << std::setprecision(10) << calculate_leary_energy(best_x, L) << std::endl;
                for(int i=0; i < best_x.rows(); i+=3) {
                    for(int j = 0; j<3; j++) {
                        if (best_x[i+j] >= 0) file << " " << std::setprecision(10) << best_x[i+j] << "\t";
                        else file << std::setprecision(10) << best_x[i+j] << "\t";
                    } 
                    file << std::endl;
                }
                file.close();
            } 
        }

        return {best_x, best_energy, calculate_leary_energy(best_x, L)};
        
    }
};