import numpy as np
import sys
import scipy.linalg as la
import matplotlib.pyplot as plt
import random
import math
import group_rep
import ham_rep
import openfermion as of
import proj
import gen_functions
import pandas as pd
from openfermion import QubitOperator
from scipy.linalg import expm
from scipy.optimize import minimize

# Initialize the Hamiltonians and the number of qubits that they act on.
nq = 4
jx = -0.5
jy = -1
jz = -1

ham_pb = of.get_sparse_operator(ham_rep.heisenberg_hamiltonian(nq, jx, jy, jz)).toarray()
ham_bpb = of.get_sparse_operator(ham_rep.break_pb(nq, jx, jy, jz)).toarray()
ham_ob = of.get_sparse_operator(ham_rep.heisenberg_hamiltonian(nq, jx, jy, jz, periodic = False)).toarray()
ham_mix = of.get_sparse_operator(ham_rep.mix_pauli('Z', nq)).toarray()

if nq == 3 and jx > 0:
	pauli_x = np.array([[0,1],[1,0]])    
	pauli_z = np.array([[1,0],[0,-1]]) 
	pauli_y = np.array([[0,-1j],[1j,0]]) 	

	spauli_x = []
	for i in range(nq):
		spauli_x.append(gen_functions.kronecker(i, nq, pauli_x))
 
	xyz = np.kron(np.kron(pauli_x, pauli_y), pauli_z) + np.kron(np.kron(pauli_z, pauli_x), pauli_y) + np.kron(np.kron(pauli_y, pauli_z), pauli_x)
 
	ham_pb = ham_pb + sum(spauli_x) + xyz 
	ham_ob = ham_ob + sum(spauli_x) + xyz

# Initial state that is used in the evolution procedure.
init_state = ham_rep.init_bra(nq, 'Z')

# Representations of the abelian groups and the projectors of the corresponding
# group.
c_group = group_rep.cyclic(nq)
proj_c = proj.projectors(c_group[0], nq)

# Test of the evolution procedure using all parameters for p = 1.
# The Hamiltonian that breaks the symmetry (ham_bpb) is used for the evolution.
gamma_1 = [i for i in list(np.arange(0, np.pi, 0.2))]

def layer_1(init_state, gamma_1):
    exph_list = []
    for i in gamma_1:
        exph1 = expm(1j*ham_bpb*i)
        exph = exph1 @ init_state.transpose()
        exph_list.append(exph)

    return exph_list 

exph_res1 = layer_1(init_state, gamma_1)

# Fidelity for each produced state with each of the projectors.
fidel_sol = []
for k, proj in enumerate(list(proj_c.values())):
    fidel = []
    for exph in exph_res1:
    	fid = exph.transpose().conj() @ proj @ exph
    	fidel.append(fid)
    fidel_sol.append(fidel)

def max_quant(params, ham_ev, layers, init_state, mix_ham, qubits):
    '''
    Function that optimizes the fidelity of the produced state, from a QAOA 
    procedure, with one of the projectors that depend on the representation of 
    the abelian group that is used. 
    Returns the objective value which is equal to the fidelity of the produced 
    state with one of the projectors.

    Args:
        params (np.array): parameters that must be optimized.
        ham_ev (np.array): Hamiltonian that is used for the evolution in the 
            QAOA procedure.
        layers (int): the number of layers of the quantum circuit.
        init_state (np.array): the initial state for the QAOA procedure.
        mix_ham (np.array): the mixing Hamiltonian that is used for QAOA.
        qubits (int): the number of qubits that all the Hamiltonians act on.
    '''
    global quantities
    global obj_val

    proj_fidel = []
    
    tmp = np.identity(2**qubits)
    beta = params[:layers]
    gamma = params[layers:]

    for i in range(layers):
        Ub = expm(-1j * mix_ham * beta[i])
        Ug = expm(-1j * ham_ev * gamma[i])
        tmp = (Ub @ Ug @ tmp)
    
    psi = tmp @ init_state
    
    for proj in proj_c.values():
        proj_fidel.append(psi.transpose().conj() @ proj @ psi)
        quantities.append(psi.transpose().conj() @ proj @ psi)

    obj = proj_fidel[3]
    obj_val.append(obj)

    return obj

# Procedure to find the local optimum parameters in order to use them as the 
# initial parameters in the QAOA procedure.
test_times = 20
p = 1
quantities = []
obj_val = []
res_init_ls = []
fun_init_ls = []
for t in range(test_times):
    init_params = np.random.uniform(-np.pi,np.pi, size = 2*p)

    res = minimize(
            fun = max_quant,
            x0 = init_params,
            args = (ham_ob, p, init_state, ham_mix, nq),
            method='COBYLA', 
            options={'maxiter': 5000})

    res_init_ls.append(res.x)
    fun_init_ls.append(res.fun)

res_init = res_init_ls[np.argmin(fun_init_ls)]

# Implementation of the QAOA procedure.
p = [i for i in range(9)]
fun = []
results = []
quant_dict = {}
for lay in p:
    quantities = []
    obj_val = []
    init_params = res_init 

    if lay > 1:
        init_params = np.random.uniform(-np.pi,np.pi, size = 2*lay)
        init_params[:len(res_tmp)] = res_tmp
    
    res = minimize(
            fun = max_quant,
            x0 = init_params,
            args = (ham_ob, lay, init_state, ham_mix, nq),
            method='COBYLA', 
            options={'maxiter': 200000})
    res_tmp = res.x
    results.append(res.x)
    fun.append(res.fun)
    
    quant_dict[lay] = {'pr': quantities, 'obj': obj_val}

    print("Fidelity:", res.fun)
    print("--Optimisation completed for p =", lay)

for i in p:
    quant_dict[i]['pr'] = [quant_dict[i]['pr'][x:x+nq] for x in range(0, len(quant_dict[i]['pr']), nq)]

best_proj = []
for lay in p:
    best_proj.append(quant_dict[lay]['pr'][np.argmin(quant_dict[lay]['obj'])])

def projectors_best_fidel(best_proj):
    '''
    Function that is used to get the fidelities that were calculated for the 
    produced states in a more convenient shape. 
    Returns a list of lists that contain the fidelities of the best state that 
    was found from the optimisation procedure for every layer of the quantum 
    circuit. Each sublist contains the fidelity of the state with each 
    projector.
    
    Args:
        best_proj (list): the fidelities of the best states that were produced 
            from the optimisation procedure for each layer. Each sublist 
            contains the fielities of a specific projector.
    '''
    proj_plt = [[] for _ in range(nq)]
    for i in range(len(proj_plt)):
        for proj in best_proj:
            for j,k in enumerate(proj):
                if i == j:
                    proj_plt[i].append(k)
                        
    return proj_plt

proj_plt = projectors_best_fidel(best_proj)

# Save the necessary data that was produced from the optimisation procedure in 
# a .csv file.
df = pd.DataFrame(proj_plt)
df.to_csv(f"data/{nq}q_proj_{sys.argv[1]}.csv", index=False)
