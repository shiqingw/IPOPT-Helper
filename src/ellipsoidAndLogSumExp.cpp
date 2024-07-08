#include "ellipsoidAndLogSumExp.hpp"

EllipsoidAndLogSumExpNLP::EllipsoidAndLogSumExpNLP(int n, const xt::xarray<double>& Q, 
            const xt::xarray<double>& mu, const xt::xarray<double>& A, const xt::xarray<double>& b, 
            double kappa)
    : Q_(Q), mu_(mu), A_(A), b_(b), kappa_(kappa), n_(n) {
    assert(Q.shape()[0] == n && Q.shape()[1] == n);
    assert(mu.shape()[0] == n);
    assert(A.shape()[0] == b.shape()[0] && A.shape()[1] == n);

    initial_guess_x_ = xt::zeros<double>({n_});
    initial_guess_lambda_ = 0.0;
    initial_guess_z_L_ = xt::zeros<double>({n_});
    initial_guess_z_U_ = xt::zeros<double>({n_});

    optimal_solution_x_ = xt::zeros<double>({n_});
    optimal_solution_lambda_ = 0.0;
    optimal_solution_z_L_ = xt::zeros<double>({n_});
    optimal_solution_z_U_ = xt::zeros<double>({n_});

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
    for (Index i = 0; i < n; ++i) {
        x[i] = initial_guess_x_(i);
    }

    if (init_z){
        for (Index i = 0; i < n; ++i) {
            z_L[i] = initial_guess_z_L_(i);
            z_U[i] = initial_guess_z_U_(i);
        }
    }

    if (init_lambda){
        lambda[0] = initial_guess_lambda_;
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
    
    for (Index i = 0; i < n; ++i) {
        optimal_solution_x_(i) = x[i];
        optimal_solution_z_L_(i) = z_L[i];
        optimal_solution_z_L_(i) = z_U[i];
        initial_guess_x_(i) = x[i];
        initial_guess_z_L_(i) = z_L[i];
        initial_guess_z_U_(i) = z_U[i];
    }
    optimal_solution_lambda_ = lambda[0];
    initial_guess_lambda_ = lambda[0];
}

EllipsoidAndLogSumExpNLPAndSolver::EllipsoidAndLogSumExpNLPAndSolver(int n, const xt::xarray<double>& Q, const xt::xarray<double>& mu,
          const xt::xarray<double>& A, const xt::xarray<double>& b, double kappa){
    nlp = initialize_nlp(n, Q, mu, A, b, kappa);
    app = initialize_solver();
}

SmartPtr<EllipsoidAndLogSumExpNLP> EllipsoidAndLogSumExpNLPAndSolver::initialize_nlp(int n, const xt::xarray<double>& Q, const xt::xarray<double>& mu,
          const xt::xarray<double>& A, const xt::xarray<double>& b, double kappa){
    nlp = new EllipsoidAndLogSumExpNLP(n, Q, mu, A, b, kappa);
    return nlp;
}

SmartPtr<IpoptApplication> EllipsoidAndLogSumExpNLPAndSolver::initialize_solver(){
    app = IpoptApplicationFactory();
    app->Options()->SetStringValue("linear_solver", "mumps");
    app->Options()->SetStringValue("sb", "yes");
    app->Options()->SetIntegerValue("print_level", 0); // Make Ipopt non-verbose
    app->Options()->SetIntegerValue("file_print_level", 0); // Optional: Also silence file output
    // app->Options()->SetIntegerValue("max_iter", 10); // Set the maximum number of iterations
    app->Initialize();
    return app;
}

void EllipsoidAndLogSumExpNLPAndSolver::update_initial_guess(const xt::xarray<double>& initial_guess_x, double initial_guess_lambda,
            const xt::xarray<double>& initial_guess_z_L, const xt::xarray<double>& initial_guess_z_U){
    assert(initial_guess_x.shape()[0] == nlp->n_);
    assert(initial_guess_z_L.shape()[0] == nlp->n_);
    assert(initial_guess_z_U.shape()[0] == nlp->n_);
    nlp->initial_guess_x_ = initial_guess_x;
    nlp->initial_guess_lambda_ = initial_guess_lambda;
    nlp->initial_guess_z_L_ = initial_guess_z_L;
    nlp->initial_guess_z_U_ = initial_guess_z_U;
}

void EllipsoidAndLogSumExpNLPAndSolver::update_problem_data(const xt::xarray<double>& Q, const xt::xarray<double>& mu,
                                     const xt::xarray<double>& A, const xt::xarray<double>& b, double kappa){
    assert(Q.shape()[0] == nlp->n_ && Q.shape()[1] == nlp->n_);
    assert(mu.shape()[0] == nlp->n_);
    assert(A.shape()[0] == b.shape()[0] && A.shape()[1] == nlp->n_);

    nlp->Q_ = Q;
    nlp->mu_ = mu;
    nlp->A_ = A;
    nlp->b_ = b;
    nlp->kappa_ = kappa;
}

void EllipsoidAndLogSumExpNLPAndSolver::solve() {
    ApplicationReturnStatus status;
    status = app->OptimizeTNLP(nlp);
    assert(status == Solve_Succeeded);
}

xt::xarray<double> EllipsoidAndLogSumExpNLPAndSolver::get_optimal_solution_x() {
    return nlp->optimal_solution_x_;
}

std::tuple<xt::xarray<double>, double, xt::xarray<double>, xt::xarray<double>> EllipsoidAndLogSumExpNLPAndSolver::get_optimal_solution() {
    return std::make_tuple(nlp->optimal_solution_x_, nlp->optimal_solution_lambda_, nlp->optimal_solution_z_L_, nlp->optimal_solution_z_U_);
}

std::tuple<xt::xarray<double>, double, xt::xarray<double>, xt::xarray<double>> EllipsoidAndLogSumExpNLPAndSolver::get_initial_guess() {
    return std::make_tuple(nlp->initial_guess_x_, nlp->initial_guess_lambda_, nlp->initial_guess_z_L_, nlp->initial_guess_z_U_);
}

EllipsoidAndLogSumExpNLPAndSolverMultiple::EllipsoidAndLogSumExpNLPAndSolverMultiple(int n_workers, const xt::xarray<double>& all_n, 
    const xt::xarray<double>& all_Q, const xt::xarray<double>& all_mu, const xt::xarray<double>& all_A, const xt::xarray<double>& all_b, 
    const xt::xarray<double>& all_kappa){
    n_workers_ = n_workers;
    for (int i = 0; i < n_workers; ++i) {
        int n = all_n(i);
        xt::xarray<double> Q = xt::view(all_Q, i, xt::all());
        xt::xarray<double> mu = xt::view(all_mu, i, xt::all());
        xt::xarray<double> A = xt::view(all_A, i, xt::all(), xt::all());
        xt::xarray<double> b = xt::view(all_b, i, xt::all());
        double kappa = all_kappa(i);

        EllipsoidAndLogSumExpNLPAndSolver* e = new EllipsoidAndLogSumExpNLPAndSolver(n, Q, mu, A, b, kappa);
        nlp_and_solvers_.push_back(e);
    }
}

EllipsoidAndLogSumExpNLPAndSolverMultiple::~EllipsoidAndLogSumExpNLPAndSolverMultiple(){
    for (int i = 0; i < n_workers_; ++i) {
        delete nlp_and_solvers_[i];
    }
}

void EllipsoidAndLogSumExpNLPAndSolverMultiple::update_initial_guess(const xt::xarray<double>& all_initial_guess_x, const xt::xarray<double>& all_initial_guess_lambda,
        const xt::xarray<double>& all_initial_guess_z_L, const xt::xarray<double>& all_initial_guess_z_U){
    for (int i = 0; i < n_workers_; ++i) {
        xt::xarray<double> initial_guess_x = xt::view(all_initial_guess_x, i, xt::all());
        double initial_guess_lambda = all_initial_guess_lambda(i);
        xt::xarray<double> initial_guess_z_L = xt::view(all_initial_guess_z_L, i, xt::all());
        xt::xarray<double> initial_guess_z_U = xt::view(all_initial_guess_z_U, i, xt::all());
        nlp_and_solvers_[i]->update_initial_guess(initial_guess_x, initial_guess_lambda, initial_guess_z_L, initial_guess_z_U);
    }
}

void EllipsoidAndLogSumExpNLPAndSolverMultiple::update_problem_data(const xt::xarray<double>& all_Q, const xt::xarray<double>& all_mu,
    const xt::xarray<double>& all_A, const xt::xarray<double>& all_b, const xt::xarray<double>& all_kappa){
    for (int i = 0; i < n_workers_; ++i) {
        xt::xarray<double> Q = xt::view(all_Q, i, xt::all(), xt::all());
        xt::xarray<double> mu = xt::view(all_mu, i, xt::all());
        xt::xarray<double> A = xt::view(all_A, i, xt::all(), xt::all());
        xt::xarray<double> b = xt::view(all_b, i, xt::all());
        double kappa = all_kappa(i);
        nlp_and_solvers_[i]->update_problem_data(Q, mu, A, b, kappa);
    }
}

void EllipsoidAndLogSumExpNLPAndSolverMultiple::solve() {
    std::vector<pid_t> pids(n_workers_);
    int pipes_x[n_workers_][2];
    int pipes_lambda[n_workers_][2];
    int pipes_z_L[n_workers_][2];
    int pipes_z_U[n_workers_][2];


    for (int i = 0; i < n_workers_; ++i) {
        if (pipe(pipes_x[i]) == -1) {
            std::cerr << "Failed to create pipe for worker " << i << std::endl;
            exit(1);
        }

        if (pipe(pipes_lambda[i]) == -1) {
            std::cerr << "Failed to create pipe for worker " << i << std::endl;
            exit(1);
        }

        if (pipe(pipes_z_L[i]) == -1) {
            std::cerr << "Failed to create pipe for worker " << i << std::endl;
            exit(1);
        }

        if (pipe(pipes_z_U[i]) == -1) {
            std::cerr << "Failed to create pipe for worker " << i << std::endl;
            exit(1);
        }

        pid_t pid = fork();
        if (pid == 0) {
            // Child process
            close(pipes_x[i][0]); // Close read end
            close(pipes_lambda[i][0]); // Close read end
            close(pipes_z_L[i][0]); // Close read end
            close(pipes_z_U[i][0]); // Close read end
            nlp_and_solvers_[i]->solve();

            // Write results to pipe
            xt::xarray<double> optimal_solution_x, optimal_solution_z_L, optimal_solution_z_U;
            double optimal_solution_lambda;
            std::tie(optimal_solution_x, optimal_solution_lambda, optimal_solution_z_L, optimal_solution_z_U) = nlp_and_solvers_[i]->get_optimal_solution();
            write(pipes_x[i][1], optimal_solution_x.data(), optimal_solution_x.size() * sizeof(double));
            write(pipes_lambda[i][1], &optimal_solution_lambda, sizeof(double));
            write(pipes_z_L[i][1], optimal_solution_z_L.data(), optimal_solution_z_L.size() * sizeof(double));
            write(pipes_z_U[i][1], optimal_solution_z_U.data(), optimal_solution_z_U.size() * sizeof(double));
            close(pipes_x[i][1]);
            close(pipes_lambda[i][1]);
            close(pipes_z_L[i][1]);
            close(pipes_z_U[i][1]);
            _exit(0); // Exit child process
        } else if (pid > 0) {
            // Parent process
            close(pipes_x[i][1]); // Close write end
            close(pipes_lambda[i][1]); // Close write end
            close(pipes_z_L[i][1]); // Close write end
            close(pipes_z_U[i][1]); // Close write end
            pids[i] = pid;
        } else {
            std::cerr << "Failed to fork process for worker " << i << std::endl;
            exit(1);
        }
    }

    // Wait for all child processes to complete and read results
    for (int i = 0; i < n_workers_; ++i) {
        int status;
        pid_t waited_pid = waitpid(pids[i], &status, 0);
        if (waited_pid == -1) {
            std::cerr << "Error waiting for process " << pids[i] << std::endl;
        } else if (WIFEXITED(status)) {
            // Read results from pipe
            read(pipes_x[i][0], nlp_and_solvers_[i]->nlp->optimal_solution_x_.data(), nlp_and_solvers_[i]->nlp->optimal_solution_x_.size() * sizeof(double));
            read(pipes_lambda[i][0], &nlp_and_solvers_[i]->nlp->optimal_solution_lambda_, sizeof(double));
            read(pipes_z_L[i][0], nlp_and_solvers_[i]->nlp->optimal_solution_z_L_.data(), nlp_and_solvers_[i]->nlp->optimal_solution_z_L_.size() * sizeof(double));
            read(pipes_z_U[i][0], nlp_and_solvers_[i]->nlp->optimal_solution_z_U_.data(), nlp_and_solvers_[i]->nlp->optimal_solution_z_U_.size() * sizeof(double));
            close(pipes_x[i][0]);
            close(pipes_lambda[i][0]);
            close(pipes_z_L[i][0]);
            close(pipes_z_U[i][0]);
        } else if (WIFSIGNALED(status)) {
            std::cerr << "Process " << waited_pid << " killed by signal " << WTERMSIG(status) << std::endl;
        }
    }
}

xt::xarray<double> EllipsoidAndLogSumExpNLPAndSolverMultiple::get_optimal_solution_x() {
    xt::xarray<double> all_optimal_solution_x = xt::zeros<double>({n_workers_, nlp_and_solvers_[0]->nlp->n_});
    for (int i = 0; i < n_workers_; ++i) {
        xt::view(all_optimal_solution_x, i, xt::all()) = nlp_and_solvers_[i]->get_optimal_solution_x();
    }
    return all_optimal_solution_x;
}

std::tuple<xt::xarray<double>, xt::xarray<double>, xt::xarray<double>, xt::xarray<double>> EllipsoidAndLogSumExpNLPAndSolverMultiple::get_optimal_solution(){
    xt::xarray<double> all_optimal_solution_x = xt::zeros<double>({n_workers_, nlp_and_solvers_[0]->nlp->n_});
    xt::xarray<double> all_optimal_solution_lambda = xt::zeros<double>({n_workers_});
    xt::xarray<double> all_optimal_solution_z_L = xt::zeros<double>({n_workers_, nlp_and_solvers_[0]->nlp->n_});
    xt::xarray<double> all_optimal_solution_z_U = xt::zeros<double>({n_workers_, nlp_and_solvers_[0]->nlp->n_});
    for (int i = 0; i < n_workers_; ++i) {
        xt::view(all_optimal_solution_x, i, xt::all()) = nlp_and_solvers_[i]->get_optimal_solution_x();
        all_optimal_solution_lambda(i) = nlp_and_solvers_[i]->nlp->optimal_solution_lambda_;
        xt::view(all_optimal_solution_z_L, i, xt::all()) = nlp_and_solvers_[i]->nlp->optimal_solution_z_L_;
        xt::view(all_optimal_solution_z_U, i, xt::all()) = nlp_and_solvers_[i]->nlp->optimal_solution_z_U_;
    }
    return std::make_tuple(all_optimal_solution_x, all_optimal_solution_lambda, all_optimal_solution_z_L, all_optimal_solution_z_U);
}

std::tuple<xt::xarray<double>, xt::xarray<double>, xt::xarray<double>, xt::xarray<double>> EllipsoidAndLogSumExpNLPAndSolverMultiple::get_initial_guess(){
    xt::xarray<double> all_initial_guess_x = xt::zeros<double>({n_workers_, nlp_and_solvers_[0]->nlp->n_});
    xt::xarray<double> all_initial_guess_lambda = xt::zeros<double>({n_workers_});
    xt::xarray<double> all_initial_guess_z_L = xt::zeros<double>({n_workers_, nlp_and_solvers_[0]->nlp->n_});
    xt::xarray<double> all_initial_guess_z_U = xt::zeros<double>({n_workers_, nlp_and_solvers_[0]->nlp->n_});
    for (int i = 0; i < n_workers_; ++i) {
        xt::xarray<double> initial_guess_x;
        double initial_guess_lambda;
        xt::xarray<double> initial_guess_z_L;
        xt::xarray<double> initial_guess_z_U;
        std::tie(initial_guess_x, initial_guess_lambda, initial_guess_z_L, initial_guess_z_U) = nlp_and_solvers_[i]->get_initial_guess();
        xt::view(all_initial_guess_x, i, xt::all()) = initial_guess_x;
        all_initial_guess_lambda(i) = initial_guess_lambda;
        xt::view(all_initial_guess_z_L, i, xt::all()) = initial_guess_z_L;
        xt::view(all_initial_guess_z_U, i, xt::all()) = initial_guess_z_U;
    }
    return std::make_tuple(all_initial_guess_x, all_initial_guess_lambda, all_initial_guess_z_L, all_initial_guess_z_U);
}