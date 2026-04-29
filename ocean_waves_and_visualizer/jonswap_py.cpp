#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "jonswap.hpp"

namespace py = pybind11;

PYBIND11_MODULE(jonswap_py, m) {
    py::enum_<JonswapConditions>(m, "JonswapConditions")
        .value("CALM",     JonswapConditions::CALM)
        .value("MODERATE", JonswapConditions::MODERATE)
        .value("STORMY",   JonswapConditions::STORMY);

    py::class_<Wave>(m, "Wave")
        .def_readonly("amplitude",    &Wave::amplitude)
        .def_readonly("wavelength",   &Wave::wavelength)
        .def_readonly("angular_freq", &Wave::angular_freq)
        .def_readonly("phase",        &Wave::phase)
        .def_readonly("dir_x",        &Wave::dir_x)
        .def_readonly("dir_y",        &Wave::dir_y);

    m.def("generate_waves", &generate_waves,
        py::arg("conditions"),
        py::arg("dominant_dir_x") = -1.0f,
        py::arg("dominant_dir_y") = -1.0f,
        py::arg("n_waves")        = NUM_WAVES,
        py::arg("seed")           = 42u);

    m.def("height_at", &height_at,
        py::arg("waves"), py::arg("x"), py::arg("y"), py::arg("t"));

    m.def("height_grid", &height_grid,
        py::arg("waves"), py::arg("xs"), py::arg("ys"), py::arg("t"));
}
