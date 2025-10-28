# LRO_DRAG-MAIN
This repository includes all the necessary code to reproduce the results of [Zhang et al. 2025](**ADD LINK TO PAPER**). In particular, this software:
1. Simulates the evolution of many 2D spin lattices, coupled via inter-layer spin-glass interactions. In the first layer only, memory variables exist which couple to the spins.
2. Visualizes some of the paper's key results (intra-layer avalanche distributions and phase diagrams).

## *main_dense_connection.py*
After downloading all *.py files in this repo, main_dense_connection.py is ready to be run from the primary directory. Be default, this will create new subdirectories "data/", "figures/", and "histograms/" in which layered spin lattice systems are simulated, avalanche distributions are extracted/plotted, and the phase structure in parameter space is visualized.

All free parameters for simulation/visualization can be added as arguments when running main_dense_connection.py. See parameter_notes.txt for a description of each potential argument.

## Contact

Email: csipling@ucsd.edu