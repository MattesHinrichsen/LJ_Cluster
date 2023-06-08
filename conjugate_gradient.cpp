#pragma once
#include"../Eigen/Sparse"
#include"../Eigen/Dense"

template<typename T>
class conjugate_gradient_optimizer {
    using Vector = Eigen::Matrix<T, -1, 1>;

    Vector x_start;
    Vector (*gradient_function)(const Vector&, T);
    Vector (*keep_in_box)(Vector, T);
    T (*objective_function)(const Vector&, T);
    T tolerance;
    T L;

    T line_search(const Vector& x, const Vector& grad_x) {
        constexpr T c = pow(10, -4);
        constexpr T tau = 0.1;
        T alpha = L * pow(tau, 2);

        T f_x = this->objective_function(x, L);

        Vector dx = x + alpha * grad_x;

        while(!( (f_x - this->objective_function(dx, L)) >= 1 * alpha * c )) {
            alpha *= tau;

            if (alpha < pow(10,-12)) {
                return 0;
            }

            dx = x + alpha * grad_x;
        }

        return alpha;        
    }

public:
    conjugate_gradient_optimizer(Vector x, T (*F)(const Vector&, T), Vector (*dF)(const Vector&, T), Vector (*keep_in_box)(Vector, T), T tolerance, T L) : x_start(x), objective_function(F), gradient_function(dF), keep_in_box(keep_in_box), tolerance(tolerance), L(L) {}


    Vector optimize() {
        Vector Delta_x = this->gradient_function(x_start, L);
        auto Delta_x_norm = Delta_x.normalized();

        T a0 = line_search(x_start, Delta_x_norm);

        Vector x = x_start + a0 * Delta_x_norm;

        x = keep_in_box(x,L);

        Vector s = Delta_x;

        Vector Delta_x_previous = Delta_x;
        Delta_x = this->gradient_function(x, L);
        

        int counter = 0;
        int alpha_equal_zero_counter = 0;
        int small_rel_error_counter = 0;

        while(true) { 

            T beta_PR = Delta_x.dot(Delta_x - Delta_x_previous) / (Delta_x_previous.dot(Delta_x_previous));
            T beta = std::max((T)0, beta_PR);

            s = Delta_x + beta * s;
            auto s_norm = s.normalized();

            T alpha = line_search(x, s_norm);
            
            if (alpha == 0) {
                alpha = line_search(x, Delta_x.normalized());
                x = x + alpha * Delta_x.normalized();
            } else {
                x = x + alpha * s_norm;
            }

            x = keep_in_box(x,L);

            if (alpha==0) {
                alpha_equal_zero_counter++;
                if (alpha_equal_zero_counter == 3) {
                    return x;
                }
            } else {
                alpha_equal_zero_counter = 0;
            }

            Delta_x_previous = Delta_x;

            Delta_x = this->gradient_function(x, L);

            counter++;

            if((Delta_x - Delta_x_previous).squaredNorm()/Delta_x.squaredNorm() < tolerance) {
                small_rel_error_counter++;
                if(small_rel_error_counter == 5) {
                    return x;
                }
            }
            
        }
        return x;
    }
};