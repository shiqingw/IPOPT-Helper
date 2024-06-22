#include <numeric>
#include <xtensor.hpp>
#include <pybind11/pybind11.h>
#define FORCE_IMPORT_ARRAY
#include <xtensor-python/pyarray.hpp>

#include "ellipsoidAndLogSumExp.hpp"

namespace py = pybind11;

PYBIND11_MODULE(IpoptHelper, m) {
    xt::import_numpy();
    m.doc() = "ipoptHelper";

    py::class_<EllipsoidAndLogSumExpNLPAndSolver>(m, "EllipsoidAndLogSumExpNLPAndSolver")
        .def(py::init<const xt::xarray<double>&, const xt::xarray<double>&, const xt::xarray<double>&,
            const xt::xarray<double>&, double, const xt::xarray<double>&, int>())
        .def("update_initial_guess", &EllipsoidAndLogSumExpNLPAndSolver::update_initial_guess)
        .def("update_problem_data", &EllipsoidAndLogSumExpNLPAndSolver::update_problem_data)
        .def("solve", &EllipsoidAndLogSumExpNLPAndSolver::solve)
        .def("get_optimal_solution", &EllipsoidAndLogSumExpNLPAndSolver::get_optimal_solution);

    // py::class_<EllipsoidAndLogSumExpNLP>(m, "EllipsoidAndLogSumExpNLP")
    //     .def(py::init<const xt::xarray<double>&, const xt::xarray<double>&, const xt::xarray<double>&,
    //         const xt::xarray<double>&, double, const xt::xarray<double>&, int>())
    //     .def("update_initial_guess", &EllipsoidAndLogSumExpNLP::update_initial_guess)
    //     .def("update_problem_data", &EllipsoidAndLogSumExpNLP::update_problem_data)
    //     .def("solve", &EllipsoidAndLogSumExpNLP::solve)
    //     .def_readwrite("Q", &EllipsoidAndLogSumExpNLP::Q_)
    //     .def_readwrite("mu", &EllipsoidAndLogSumExpNLP::mu_)
    //     .def_readwrite("A", &EllipsoidAndLogSumExpNLP::A_)
    //     .def_readwrite("b", &EllipsoidAndLogSumExpNLP::b_)
    //     .def_readwrite("kappa", &EllipsoidAndLogSumExpNLP::kappa_)
    //     .def_readwrite("initial_guess", &EllipsoidAndLogSumExpNLP::initial_guess_)
    //     .def_readwrite("optimal_solution", &EllipsoidAndLogSumExpNLP::optimal_solution_);


}
