# Moving to a new eigenspace with QAOA

Python files that contain the necessary functions in order to find the minimum number of layers needed to produce a state (from an 
optimisation procedure like Quantum Approximate Optimisation Algorithm) that will completely belong to an eigenspace different 
than the eigenspace of the initial state.

## Description of files

The main file of the specific project is the [high_quant.py](/high_quant.py). It contains the code for the main experiment. The
results are stored in a .csv file in the [data](/data) folder.

All the other python files contain supplementary functions that are necessary in order to successfully conduct the main 
experiment. The [qaoa_opt.py](/qaoa_opt.py) contains the code for the implementation of a QAOA procedure and the 
[create_plt.py](/create_plt.py) is used in order to create the plots that derive from the data we took from the 
experiments. 
