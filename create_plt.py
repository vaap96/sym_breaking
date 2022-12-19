import csv
import matplotlib.pyplot as plt
import sys
import pandas as pd
import group_rep
import numpy as np
import proj

opt_type = sys.argv[1]
nq = int(sys.argv[2])
p = [i for i in range(9)]

# Define Groups and Projectors
cycle = group_rep.cyclic(nq)[0]
proj_c = proj.projectors(cycle, nq)

# Load the data files and store them to lists in order plot the results.
df = pd.read_csv(f"data/{nq}q_proj_{opt_type}.csv")
proj_plt = df.values.tolist()
for i in range(len(proj_plt)):
    proj_plt[i] = list(map(complex, proj_plt[i]))

# Plot how the best energy and fidelity that were found evolves when p is 
# increased -- Fidelity of projectors is also plotted.
plt.rc('font', family='serif')
plt.rcParams.update({'font.size': 18})   
plt.rcParams['axes.labelweight'] = 'bold'
colours = ['red','brown','blue','cyan','orange','black','olive','purple']
markers = ['>', 's', 'o', '.', '^', 'p', 'D', '8']

def quant_plt(col, mark, proj, p, n_qubits, filename):
    '''
    Function that is used to plot the fidelities of all the projectors with the 
    produced states for every layer. Only the fidelity of the optimal state is 
    plotted for each layer.

    Args:
        col (list): colours that will be used in the plots.
        mark (list): markers that will be used in the plots.
        proj (list): the values of fidelity that will be plotted for all 
            projectors.
        n_qubits (int): the number of qubits that the Hamiltonians act on.
        filename (str): the name of the file that will be created.
    '''
    fig, axs = plt.subplots(1, 1, figsize=(20,8))    

    axs.set_yscale('log')
    for i, k in enumerate(proj):
        axs.plot(p, k, color = col[i], marker = mark[i], linewidth = 3,  
                label=f'Projector {list(proj_c.keys())[i].round(2)}')
    axs.set_ylabel('Fidelity')
    axs.set_xlabel('Layers')
    axs.legend()
    
    plt.savefig(filename)
    plt.close()

quant_plt(colours, markers, proj_plt, p, nq, 'plots/proj_test_log_max.png')
