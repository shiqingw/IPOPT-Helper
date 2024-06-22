#include <cassert>
#include <iostream>
#include <vector>
#include <cmath>
#include "IpIpoptApplication.hpp"
#include "IpSolveStatistics.hpp"
#include "IpTNLP.hpp"

using namespace Ipopt;

class MyNLP : public TNLP {
public:
    MyNLP(const std::vector<double>& Q, const std::vector<double>& mu,
          const std::vector<double>& A, const std::vector<double>& b, double kappa,
          const std::vector<double>& initial_guess, int n, int m)
        : Q_(Q), mu_(mu), A_(A), b_(b), kappa_(kappa), initial_guess_(initial_guess), n_(n), m_(m) {
        assert(Q.size() == n * n);
        assert(mu.size() == n);
        assert(A.size() == m * n);
        assert(b.size() == m);
        assert(initial_guess.size() == n);
    }

    virtual bool get_nlp_info(Index& n, Index& m, Index& nnz_jac_g, Index& nnz_h_lag, 
                              IndexStyleEnum& index_style) {
        n = n_; // Number of variables
        m = 1;  // Number of constraints (logsumexp)
        nnz_jac_g = n_; // Non-zeros in the Jacobian (logsumexp gradient)
        nnz_h_lag = n_ * (n_ + 1) / 2; // Non-zeros in the Hessian
        index_style = TNLP::C_STYLE;
        return true;
    }

    virtual bool get_bounds_info(Index n, Number* x_l, Number* x_u, 
                                 Index m, Number* g_l, Number* g_u) {
        for (Index i = 0; i < n; ++i) {
            x_l[i] = -1e19;
            x_u[i] = 1e19;
        }
        g_l[0] = -1e19;
        g_u[0] = 1.0; // logsumexp <= 1
        return true;
    }

    virtual bool get_starting_point(Index n, bool init_x, Number* x, 
                                    bool init_z, Number* z_L, Number* z_U, 
                                    Index m, bool init_lambda, Number* lambda) {
        assert(init_x == true);
        for (Index i = 0; i < n; ++i) {
            x[i] = initial_guess_[i];
        }
        return true;
    }

    virtual bool eval_f(Index n, const Number* x, bool new_x, Number& obj_value) {
        obj_value = 0.0;
        for (Index i = 0; i < n; ++i) {
            for (Index j = 0; j < n; ++j) {
                obj_value += Q_[i * n + j] * (x[i] - mu_[i]) * (x[j] - mu_[j]);
            }
        }
        return true;
    }

    virtual bool eval_grad_f(Index n, const Number* x, bool new_x, Number* grad_f) {
        for (Index i = 0; i < n; ++i) {
            grad_f[i] = 0.0;
            for (Index j = 0; j < n; ++j) {
                grad_f[i] += Q_[i * n + j] * (x[j] - mu_[j]) + Q_[j * n + i] * (x[j] - mu_[i]);
            }
        }
        return true;
    }

    virtual bool eval_g(Index n, const Number* x, bool new_x, Index m, Number* g) {
        g[0] = 0.0;
        for (Index i = 0; i < m_; ++i) {
            double sum = 0.0;
            for (Index j = 0; j < n_; ++j) {
                sum += A_[i * n_ + j] * x[j];
            }
            sum += b_[i];
            g[0] += std::exp(kappa_ * sum);
        }
        g[0] = std::log(g[0]);
        return true;
    }

    virtual bool eval_jac_g(Index n, const Number* x, bool new_x, Index m, Index nele_jac, 
                            Index* iRow, Index* jCol, Number* values) {
        if (values == NULL) {
            for (Index j = 0; j < n_; ++j) {
                iRow[j] = 0;
                jCol[j] = j;
            }
        } else {
            double sum_g = 0.0;
            std::vector<double> exp_terms(m_);
            for (Index i = 0; i < m_; ++i) {
                double sum = 0.0;
                for (Index j = 0; j < n_; ++j) {
                    sum += A_[i * n_ + j] * x[j];
                }
                sum += b_[i];
                exp_terms[i] = std::exp(kappa_ * sum);
                sum_g += exp_terms[i];
            }
            for (Index j = 0; j < n_; ++j) {
                values[j] = 0.0;
                for (Index i = 0; i < m_; ++i) {
                    values[j] += kappa_ * A_[i * n_ + j] * exp_terms[i];
                }
                values[j] /= sum_g;
            }
        }
        return true;
    }

    virtual bool eval_h(Index n, const Number* x, bool new_x, Number obj_factor, 
                        Index m, const Number* lambda, bool new_lambda, 
                        Index nele_hess, Index* iRow, Index* jCol, Number* values) {
        if (values == NULL) {
            Index idx = 0;
            for (Index i = 0; i < n_; ++i) {
                for (Index j = 0; j <= i; ++j) {
                    iRow[idx] = i;
                    jCol[idx] = j;
                    ++idx;
                }
            }
        } else {
            Index idx = 0;
            for (Index i = 0; i < n_; ++i) {
                for (Index j = 0; j <= i; ++j) {
                    values[idx] = obj_factor * (Q_[i * n_ + j] + Q_[j * n_ + i]);
                    ++idx;
                }
            }
        }
        return true;
    }

    virtual void finalize_solution(SolverReturn status, Index n, const Number* x, 
                                   const Number* z_L, const Number* z_U, 
                                   Index m, const Number* g, const Number* lambda, 
                                   Number obj_value, const IpoptData* ip_data, 
                                   IpoptCalculatedQuantities* ip_cq) {
        std::cout << "Solution of the primal variables: " << std::endl;
        for (Index i = 0; i < n; i++) {
            std::cout << "x[" << i << "] = " << x[i] << std::endl;
        }
        std::cout << "Objective value: " << obj_value << std::endl;
    }

private:
    std::vector<double> Q_;
    std::vector<double> mu_;
    std::vector<double> A_;
    std::vector<double> b_;
    double kappa_;
    std::vector<double> initial_guess_;
    int n_;
    int m_;
};

int main(int argc, char* argv[]) {
    // Define the problem dimensions
    int n = 3; // Number of variables
    int m = 2; // Number of constraints (rows in A)

    // Define problem parameters
    std::vector<double> Q = {1, 0, 0, 0, 1, 0, 0, 0, 1}; // Q matrix (3x3)
    std::vector<double> mu = {1, 1, 1}; // mu vector (3x1)
    std::vector<double> A = {1, 2, 3, 4, 5, 6}; // A matrix (2x3)
    std::vector<double> b = {1, 2}; // b vector (2x1)
    double kappa = 1.0;
    std::vector<double> initial_guess = {0.5, 0.5, 0.5}; // Initial guess

    // Create an instance of the IpoptApplication
    SmartPtr<IpoptApplication> app = IpoptApplicationFactory();
    app->Options()->SetStringValue("linear_solver", "mumps");
    app->Options()->SetIntegerValue("max_iter", 100); // Set the maximum number of iterations
    app->Initialize();

    // Create an instance of the problem
    SmartPtr<TNLP> mynlp = new MyNLP(Q, mu, A, b, kappa, initial_guess, n, m);

    // Optimize
    ApplicationReturnStatus status = app->OptimizeTNLP(mynlp);

    if (status == Solve_Succeeded) {
        std::cout << "The problem solved!" << std::endl;
    } else {
        std::cout << "The problem failed!" << std::endl;
    }

    return (int) status;
}
