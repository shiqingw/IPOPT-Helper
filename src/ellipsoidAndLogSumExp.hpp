#ifndef ELLIPSOID_AND_LOG_SUM_EXP_HPP
#define ELLIPSOID_AND_LOG_SUM_EXP_HPP

#include <cassert>
#include <iostream>
#include <vector>
#include <cmath>
#include <sys/wait.h>
#include <unistd.h>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xadapt.hpp>
#include <IpIpoptApplication.hpp>
#include <IpSolveStatistics.hpp>
#include <IpTNLP.hpp>

using namespace Ipopt;

class EllipsoidAndLogSumExpNLP : public TNLP {
public:
    int n_;
    xt::xarray<double> Q_;
    xt::xarray<double> mu_;
    xt::xarray<double> A_;
    xt::xarray<double> b_;
    double kappa_;
    xt::xarray<double> initial_guess_x_;
    double initial_guess_lambda_;
    xt::xarray<double> initial_guess_z_L_;
    xt::xarray<double> initial_guess_z_U_;
    xt::xarray<double> optimal_solution_x_;
    double optimal_solution_lambda_;
    xt::xarray<double> optimal_solution_z_L_;
    xt::xarray<double> optimal_solution_z_U_;

    EllipsoidAndLogSumExpNLP(int n, const xt::xarray<double>& Q, const xt::xarray<double>& mu,
        const xt::xarray<double>& A, const xt::xarray<double>& b, double kappa);
    ~EllipsoidAndLogSumExpNLP() = default;

    bool get_nlp_info(Index& n, Index& m, Index& nnz_jac_g, Index& nnz_h_lag, 
        IndexStyleEnum& index_style) override;

    bool get_bounds_info(Index n, Number* x_l, Number* x_u, 
        Index m, Number* g_l, Number* g_u) override;

    bool get_starting_point(Index n, bool init_x, Number* x, 
        bool init_z, Number* z_L, Number* z_U, 
        Index m, bool init_lambda, Number* lambda) override;

    bool eval_f(Index n, const Number* x, bool new_x, Number& obj_value) override;

    bool eval_grad_f(Index n, const Number* x, bool new_x, Number* grad_f) override;

    bool eval_g(Index n, const Number* x, bool new_x, Index m, Number* g) override;

    bool eval_jac_g(Index n, const Number* x, bool new_x, Index m, Index nele_jac, 
        Index* iRow, Index *jCol, Number* values) override;

    bool eval_h(Index n, const Number* x, bool new_x, Number obj_factor, Index m, 
        const Number* lambda, bool new_lambda, Index nele_hess, Index* iRow,
        Index* jCol, Number* values) override;

    void finalize_solution(SolverReturn status, Index n, const Number* x, const 
        Number* z_L, const Number* z_U, Index m, const Number* g, 
        const Number* lambda, Number obj_value, const IpoptData* ip_data, 
        IpoptCalculatedQuantities* ip_cq) override;
};

class EllipsoidAndLogSumExpNLPAndSolver {
public:
    SmartPtr<EllipsoidAndLogSumExpNLP> nlp;
    SmartPtr<IpoptApplication> app;

    EllipsoidAndLogSumExpNLPAndSolver(int n, const xt::xarray<double>& Q, const xt::xarray<double>& mu,
        const xt::xarray<double>& A, const xt::xarray<double>& b, double kappa);
    ~EllipsoidAndLogSumExpNLPAndSolver() = default;

    SmartPtr<EllipsoidAndLogSumExpNLP> initialize_nlp(int n, const xt::xarray<double>& Q, const xt::xarray<double>& mu,
        const xt::xarray<double>& A, const xt::xarray<double>& b, double kappa);

    SmartPtr<IpoptApplication> initialize_solver();

    void update_initial_guess(const xt::xarray<double>& initial_guess_x, double initial_guess_lambda,
        const xt::xarray<double>& initial_guess_z_L, const xt::xarray<double>& initial_guess_z_U);

    void update_problem_data(const xt::xarray<double>& Q, const xt::xarray<double>& mu,
        const xt::xarray<double>& A, const xt::xarray<double>& b, double kappa);

    void solve();

    xt::xarray<double> get_optimal_solution_x();

    std::tuple<xt::xarray<double>, double, xt::xarray<double>, xt::xarray<double>> get_optimal_solution();

    std::tuple<xt::xarray<double>, double, xt::xarray<double>, xt::xarray<double>> get_initial_guess();

private:
    int ref_count_;
};

class EllipsoidAndLogSumExpNLPAndSolverMultiple {
public:
    int n_workers_;
    std::vector<EllipsoidAndLogSumExpNLPAndSolver*> nlp_and_solvers_;

    EllipsoidAndLogSumExpNLPAndSolverMultiple(int n_workers, const xt::xarray<double>& all_n, const xt::xarray<double>& all_Q,
        const xt::xarray<double>& all_mu, const xt::xarray<double>& all_A, const xt::xarray<double>& all_b, const xt::xarray<double>& all_kappa);

    ~EllipsoidAndLogSumExpNLPAndSolverMultiple();

    void update_initial_guess(const xt::xarray<double>& all_initial_guess_x, const xt::xarray<double>& all_initial_guess_lambda,
        const xt::xarray<double>& all_initial_guess_z_L, const xt::xarray<double>& all_initial_guess_z_U);

    void update_problem_data(const xt::xarray<double>& all_Q, const xt::xarray<double>& all_mu,
        const xt::xarray<double>& all_A, const xt::xarray<double>& all_b, const xt::xarray<double>& all_kappa);

    void solve();

    xt::xarray<double> get_optimal_solution_x();

    std::tuple<xt::xarray<double>, xt::xarray<double>, xt::xarray<double>, xt::xarray<double>> get_optimal_solution();

    std::tuple<xt::xarray<double>, xt::xarray<double>, xt::xarray<double>, xt::xarray<double>> get_initial_guess();

};

#endif // ELLIPSOID_AND_LOG_SUM_EXP_HPP