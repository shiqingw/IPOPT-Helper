#ifndef ELLIPSOID_AND_LOG_SUM_EXP_HPP
#define ELLIPSOID_AND_LOG_SUM_EXP_HPP

#include <cassert>
#include <iostream>
#include <vector>
#include <cmath>
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
    xt::xarray<double> Q_;
    xt::xarray<double> mu_;
    xt::xarray<double> A_;
    xt::xarray<double> b_;
    double kappa_;
    xt::xarray<double> initial_guess_;
    int n_;
    xt::xarray<double> optimal_solution_;

    EllipsoidAndLogSumExpNLP(const xt::xarray<double>& Q, const xt::xarray<double>& mu,
          const xt::xarray<double>& A, const xt::xarray<double>& b, double kappa,
          const xt::xarray<double>& initial_guess, int n);
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

    EllipsoidAndLogSumExpNLPAndSolver(const xt::xarray<double>& Q, const xt::xarray<double>& mu,
          const xt::xarray<double>& A, const xt::xarray<double>& b, double kappa,
          const xt::xarray<double>& initial_guess, int n);
    ~EllipsoidAndLogSumExpNLPAndSolver() = default;

    void update_initial_guess(const xt::xarray<double>& initial_guess);

    void update_problem_data(const xt::xarray<double>& Q, const xt::xarray<double>& mu,
                                     const xt::xarray<double>& A, const xt::xarray<double>& b, double kappa);

    void solve();
    
    xt::xarray<double> get_optimal_solution();

};

#endif // ELLIPSOID_AND_LOG_SUM_EXP_HPP