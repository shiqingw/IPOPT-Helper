#include <numeric>
#include <xtensor.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#define FORCE_IMPORT_ARRAY
#include <xtensor-python/pyarray.hpp>

#include "ellipsoidAndLogSumExp.hpp"

namespace py = pybind11;

PYBIND11_MODULE(IpoptHelper, m) {
    xt::import_numpy();
    m.doc() = "ipoptHelper";

    py::class_<EllipsoidAndLogSumExpNLPAndSolver>(m, "EllipsoidAndLogSumExpNLPAndSolver")
        .def(py::init<int, const xt::xarray<double>&, const xt::xarray<double>&, const xt::xarray<double>&,
            const xt::xarray<double>&, double>())
        .def("update_initial_guess", &EllipsoidAndLogSumExpNLPAndSolver::update_initial_guess)
        .def("update_problem_data", &EllipsoidAndLogSumExpNLPAndSolver::update_problem_data)
        .def("solve", &EllipsoidAndLogSumExpNLPAndSolver::solve)
        .def("get_optimal_solution_x", &EllipsoidAndLogSumExpNLPAndSolver::get_optimal_solution_x)
        .def("get_optimal_solution", &EllipsoidAndLogSumExpNLPAndSolver::get_optimal_solution)
        .def("get_initial_guess", &EllipsoidAndLogSumExpNLPAndSolver::get_initial_guess)
        .def(py::pickle(
            [](const EllipsoidAndLogSumExpNLPAndSolver &e) { // __getstate__
                return py::make_tuple(e.nlp->n_, e.nlp->Q_, e.nlp->mu_, e.nlp->A_, e.nlp->b_, e.nlp->kappa_, 
                                    e.nlp->optimal_solution_x_, e.nlp->optimal_solution_lambda_, 
                                    e.nlp->optimal_solution_z_L_, e.nlp->optimal_solution_z_U_);
            },
            [](py::tuple t) { // __setstate__
                if (t.size() != 10) {
                    throw std::runtime_error("Invalid state!");
                }
                EllipsoidAndLogSumExpNLPAndSolver e(t[0].cast<int>(), t[1].cast<xt::xarray<double>>(),
                    t[2].cast<xt::xarray<double>>(), t[3].cast<xt::xarray<double>>(), t[4].cast<xt::xarray<double>>(),
                    t[5].cast<double>());
                e.update_initial_guess(t[6].cast<xt::xarray<double>>(), t[7].cast<double>(), 
                    t[8].cast<xt::xarray<double>>(),
                    t[9].cast<xt::xarray<double>>());
                return e;
            }
        ));

    py::class_<EllipsoidAndLogSumExpNLPAndSolverMultiple>(m, "EllipsoidAndLogSumExpNLPAndSolverMultiple")
        .def(py::init<int, const xt::xarray<double>&, const xt::xarray<double>&, const xt::xarray<double>&, const xt::xarray<double>&,
            const xt::xarray<double>&, const xt::xarray<double>&>())
        .def("update_initial_guess", &EllipsoidAndLogSumExpNLPAndSolverMultiple::update_initial_guess)
        .def("update_problem_data", &EllipsoidAndLogSumExpNLPAndSolverMultiple::update_problem_data)
        .def("solve", &EllipsoidAndLogSumExpNLPAndSolverMultiple::solve)
        .def("get_optimal_solution_x", &EllipsoidAndLogSumExpNLPAndSolverMultiple::get_optimal_solution_x)
        .def("get_optimal_solution", &EllipsoidAndLogSumExpNLPAndSolverMultiple::get_optimal_solution)
        .def("get_initial_guess", &EllipsoidAndLogSumExpNLPAndSolverMultiple::get_initial_guess);
}