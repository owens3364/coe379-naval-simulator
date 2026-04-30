# Ocean Waves Simulation and Visualization

This directory contains code which can be used to mathematically simulate ocean waves over time using the JONSWAP wave spectrum. It additionally contains code for visualizing this simulation.

## Contents

- `gen_waves.cpp` generates a file named `waves.txt` that lists the number of waves and the JONSWAP-derived wave properties (amplitude, wavelength, angular frequency, phase shift, direction x/y) for each of them, separated by newlines.
- `jonswap_mpi.cpp` parallelizes the ocean surface computation with MPI and generates `ocean.bin` which is a binary file with a 64-byte header that contains the number of rows, cols, frames, fps, and grid dims (meters) and its body is each ocean heights matrix written directly into the file as f32s.
- `jonswap_py.cpp` provides C++ bindings so `jonswap.hpp` can be used by the Python visualizer.
- `jonswap.hpp` is a single header file that provides realistic ocean surface simulations using sampling from the JONSWAP spectrum.
- `ship_motion.cpp` simulates the boat motion and outputs a `boat_motion.txt` which can be fed (alongside ocean.bin) into the [render_simulation](../hull_mesh_generation/render_simulation.py) python script.
- `visualizer.py` visualizes a single rendering of the ocean using those bindings and `jonswap.hpp`.

## Compiling and Running

- Open a terminal in this directory and run `uv sync`. You may need to install [uv](https://docs.astral.sh/uv/) to continue.
- Compile the C++ library, making any changes to adapt the command to your compilation toolchain as appropriate (this command successfully runs on Mac computers with Apple Silicon):

```bash
c++ -O2 -shared -fPIC -Wl,-undefined,dynamic_lookup $(uv run python3 -m pybind11 --includes) jonswap_py.cpp -o jonswap_py$(uv run python3-config --extension-suffix) -std=c++17
```

- Run `uv run python visualizer.py` to start the visualizer. Any changes made to system parameters must be made in `jonswap.hpp` and then recompiled before the visualizer can be rerun to use them.
