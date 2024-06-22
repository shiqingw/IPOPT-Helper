#include "ellipsoidAndLogSumExp.hpp"

EllipsoidAndLogSumExpNLP::EllipsoidAndLogSumExpNLP(const xt::xarray<double>& Q, const xt::xarray<double>& mu,
          const xt::xarray<double>& A, const xt::xarray<double>& b, double kappa,
          const xt::xarray<double>& initial_guess, int n)
    : Q_(Q), mu_(mu), A_(A), b_(b), kappa_(kappa), initial_guess_(initial_guess), n_(n) {
    assert(Q.shape()[0] == n && Q.shape()[1] == n);
    assert(mu.shape()[0] == n);
    assert(A.shape()[0] == b.shape()[0] && A.shape()[1] == n);
    assert(initial_guess.shape()[0] == n);

    optimal_solution_ = xt::zeros<double>({n_});

    app_ = IpoptApplicationFactory();
    app_->Options()->SetStringValue("linear_solver", "mumps");
    // app_->Options()->SetIntegerValue("max_iter", 100); // Set the maximum number of iterations
    app_->Initialize();
}

void EllipsoidAndLogSumExpNLP::update_initial_guess(const xt::xarray<double>& initial_guess) {
    assert(initial_guess.shape()[0] == n_);
    initial_guess_ = initial_guess;
}

void EllipsoidAndLogSumExpNLP::update_problem_data(const xt::xarray<double>& Q, const xt::xarray<double>& mu,
                                     const xt::xarray<double>& A, const xt::xarray<double>& b, double kappa){
    assert(Q.shape()[0] == n_ && Q.shape()[1] == n_);
    assert(mu.shape()[0] == n_);
    assert(A.shape()[0] == b.shape()[0] && A.shape()[1] == n_);

    Q_ = Q;
    mu_ = mu;
    A_ = A;
    b_ = b;
    kappa_ = kappa;
}

bool EllipsoidAndLogSumExpNLP::get_nlp_info(Index& n, Index& m, Index& nnz_jac_g, Index& nnz_h_lag, 
                         IndexStyleEnum& index_style) {
    n = n_; // Number of variables
    m = 1;  // Number of constraints (logsumexp)
    nnz_jac_g = n_; // Non-zeros in the Jacobian (logsumexp gradient)
    nnz_h_lag = n_ * (n_ + 1) / 2; // Non-zeros in the Hessian
    index_style = TNLP::C_STYLE;
    return true;
}

bool EllipsoidAndLogSumExpNLP::get_bounds_info(Index n, Number* x_l, Number* x_u, 
                            Index m, Number* g_l, Number* g_u) {
    for (Index i = 0; i < n; ++i) {
        x_l[i] = -1e19;
        x_u[i] = 1e19;
    }
    g_l[0] = -1e19;
    g_u[0] = 1.0; // logsumexp <= 1
    return true;
}

bool EllipsoidAndLogSumExpNLP::get_starting_point(Index n, bool init_x, Number* x, 
                               bool init_z, Number* z_L, Number* z_U, 
                               Index m, bool init_lambda, Number* lambda) {
    assert(init_x == true);
    assert(init_z == false);
    assert(init_lambda == false);
    for (Index i = 0; i < n; ++i) {
        x[i] = initial_guess_(i);
    }
    return true;
}

bool EllipsoidAndLogSumExpNLP::eval_f(Index n, const Number* x, bool new_x, Number& obj_value) {
    xt::xarray<double> x_ = xt::zeros<double>({n_});
    for (Index i = 0; i < n; ++i) {
        x_(i) = x[i];
    }
    obj_value = xt::linalg::dot(x_ - mu_, xt::linalg::dot(Q_, x_ - mu_))(0);
    return true;
}

bool EllipsoidAndLogSumExpNLP::eval_grad_f(Index n, const Number* x, bool new_x, Number* grad_f) {
    xt::xarray<double> x_ = xt::zeros<double>({n_});
    for (Index i = 0; i < n; ++i) {
        x_(i) = x[i];
    }
    xt::xarray<double> grad_f_ = 2*xt::linalg::dot(Q_, x_ - mu_);
    for (Index i = 0; i < n; ++i) {
        grad_f[i] = grad_f_(i);
    }
    return true;
}

bool EllipsoidAndLogSumExpNLP::eval_g(Index n, const Number* x, bool new_x, Index m, Number* g) {
    xt::xarray<double> x_ = xt::zeros<double>({n_});
    for (Index i = 0; i < n; ++i) {
        x_(i) = x[i];
    }
    int dim_z = A_.shape()[0];
    xt::xarray<double> z = kappa_ * (xt::linalg::dot(A_, x_) + b_);
    double c = xt::amax(z)();
    z = xt::exp(z - c);
    double sum_z = xt::sum(z)();
    g[0] = std::log(sum_z) + c - std::log((double)dim_z) + 1;

    return true;
}

bool EllipsoidAndLogSumExpNLP::eval_jac_g(Index n, const Number* x, bool new_x, Index m, Index nele_jac, 
                       Index* iRow, Index *jCol, Number* values) {
    if (values == NULL) {
        for (Index i = 0; i < n; ++i) {
            iRow[i] = 0;
            jCol[i] = i;
        }
    } else {
        xt::xarray<double> x_ = xt::zeros<double>({n_});
        for (Index i = 0; i < n; ++i) {
            x_(i) = x[i];
        }
        int dim_z = A_.shape()[0];
        xt::xarray<double> z = kappa_ * (xt::linalg::dot(A_, x_) + b_);
        double c = xt::amax(z)();
        z = xt::exp(z - c);
        double sum_z = xt::sum(z)();

        xt::xarray<double> zT_A = xt::linalg::dot(z, A_);
        xt::xarray<double> F_dp = kappa_ * zT_A / sum_z;
        for (Index i = 0; i < n; ++i) {
            values[i] = F_dp(i);
        }
    }
    return true;
}

bool EllipsoidAndLogSumExpNLP::eval_h(Index n, const Number* x, bool new_x, Number obj_factor, Index m, 
                   const Number* lambda, bool new_lambda, Index nele_hess, Index* iRow,
                   Index* jCol, Number* values) {
    if (values == NULL) {
        Index idx = 0;
        for (Index i = 0; i < n; ++i) {
            for (Index j = 0; j <= i; ++j) {
                iRow[idx] = i;
                jCol[idx] = j;
                idx++;
            }
        }
    } else {
        xt::xarray<double> x_ = xt::zeros<double>({n_});
        for (Index i = 0; i < n; ++i) {
            x_(i) = x[i];
        }
        xt::xarray<double> z = kappa_ * (xt::linalg::dot(A_, x_) + b_);
        double c = xt::amax(z)();
        z = xt::exp(z - c);
        double sum_z = xt::sum(z)();

        xt::xarray<double> zT_A = xt::linalg::dot(z, A_);
        xt::xarray<double> diag_z = xt::diag(z);
        xt::xarray<double> diag_z_A = xt::linalg::dot(diag_z, A_);
        xt::xarray<double> AT_diag_z_A = xt::linalg::dot(xt::transpose(A_, {1,0}), diag_z_A);
        xt::xarray<double> AT_z_zT_A = xt::linalg::outer(zT_A, zT_A);
        xt::xarray<double> F_dpdp = std::pow(kappa_,2) * (AT_diag_z_A/sum_z - AT_z_zT_A/std::pow(sum_z,2));

        Index idx = 0;
        for (Index i = 0; i < n; ++i) {
            for (Index j = 0; j <= i; ++j) {
                values[idx] = Q_(i,j)+lambda[0]*F_dpdp(i, j);
                idx++;
            }
        }
    }
    return true;
}

void EllipsoidAndLogSumExpNLP::finalize_solution(SolverReturn status, Index n, const Number* x, const 
                              Number* z_L, const Number* z_U, Index m, const Number* g, 
                              const Number* lambda, Number obj_value, const IpoptData* ip_data, 
                              IpoptCalculatedQuantities* ip_cq) {
    
    optimal_solution_ = xt::zeros<double>({n_});
    for (Index i = 0; i < n; ++i) {
        optimal_solution_(i) = x[i];
    }
}

void EllipsoidAndLogSumExpNLP::solve() {
    ApplicationReturnStatus status;
    status = app_->OptimizeTNLP(this);
    assert(status == Solve_Succeeded);
}