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

# Initial state that is used in the evolution procedure.
init_state = ham_rep.init_bra(nq, 'Z')

# Representations of the abelian groups and the projectors of the corresponding
# group.
c_group = group_rep.cyclic(nq)
proj_c = proj.projectors(c_group[0], nq)

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

    obj = proj_fidel[0]
    obj_val.append(obj)

    return -obj

test_times = 20
p = 3
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
    print(res.fun)

res_init = res_init_ls[np.argmin(fun_init_ls)]
