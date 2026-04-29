# Contents

- jonswap_py.cpp is cpp bindings so jonswap.hpp can be used by Python
- visualizer.py visualizes a single rendering of the ocean using those bindings and jonswap.hpp
- jonswap.hpp is a single header file that provides realistic ocean surface sims using sampling from the jonswap spectrum

# Compiling

- Somehow you need matplotlib, numpy, and pybind11
- Compile the c++ library for your system
  - For me, the command is `c++ -O2 -shared -fPIC -Wl,-undefined,dynamic_lookup $(python -m pybind11 --includes) jonswap_py.cpp -o jonswap_py$(python-config --extension-suffix) -std=c++17`

# Running

`python visualizer.py`
