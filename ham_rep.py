import numpy as np
import sys
from openfermion import QubitOperator


def heisenberg_hamiltonian(n_qubits, jx, jy, jz, periodic=True):
    
	'''
    Generate an instance of the Heisenberg Hamiltonian.
    
    If jx = jy = jz, then XXX Hamiltonian.
    If jx = jy != jz, then XXZ Hamiltonian.
    if jx != jy != jz then XYZ Hamiltonian.
    
    Args:
        n_qubits (int): Number of qubits.
        jx (float): Strength in X-direction.
        jy (float): Strength in Y-direction.
        jz (float): Strength in Z-direction.
        periodic (bool): Periodic boundary conditions or not.
    '''
    
	hamiltonian = QubitOperator()

	# Hamiltonian with periodic boundaries.
	if periodic:
		for i in range(n_qubits):
			hamiltonian += (
				QubitOperator(f'X{i} X{(i+1)%n_qubits}', jx) +
				QubitOperator(f'Y{i} Y{(i+1)%n_qubits}', jy) +
				QubitOperator(f'Z{i} Z{(i+1)%n_qubits}', jz))
	
	# Hamiltonian with open boundaries.
	else:
		for i in range(n_qubits-1):
			hamiltonian += (
				QubitOperator(f'X{i} X{(i+1)%n_qubits}', jx) +
				QubitOperator(f'Y{i} Y{(i+1)%n_qubits}', jy) +
				QubitOperator(f'Z{i} Z{(i+1)%n_qubits}', jz))

	return hamiltonian


def break_degen(n_qubits):

	'''
	Function that is used when the ground state is degenerate. Creates a term of 
	local X and Y gates that is added to the initial Hamiltonian in order to 
	break its degeneracy.
	
	Args:
		n_qubits (int): number of qubits that the Hamiltonian acts on.
	'''

	hamiltonian = QubitOperator()
	for i in range(n_qubits):
		hamiltonian += QubitOperator(f'X{i} Y{(i+1)%n_qubits}')

	return hamiltonian


def mix_pauli(pauli, qubits):
	
	'''
	Function that generates an instance of mixing Hamiltonians that is used in 
	the QAOA procedure. 

	Args:
		pauli (str): the pauli matrix that we want to use.
		qubits (int): number of qubits that the problem Hamiltonian acts on.
	'''

	spauli = QubitOperator()
	for i in range(qubits):
		spauli += QubitOperator(f'{pauli}{i}', 1)

	return spauli

def init_bra(qubits, mix_ham):
	
	'''
	Function that generates an initial state depending on the mixing Hamiltonian
	that is used. 

	Args:
		qubits (int): number of qubits that the problem Hamiltonian acts on.
		mix_ham (str): the Pauli gate that is used for the mixing Hamiltonian.
	'''

	if mix_ham == 'X':
		init = -1/math.sqrt(2) * np.array([1,1])
		minus = 1/math.sqrt(2) * np.array([1,1])
		for i in range(1, qubits):
			init = np.kron(init, minus)
	elif mix_ham == '+-':
		init = 1/math.sqrt(2) * np.array([1,1])
		minus = 1/math.sqrt(2) * np.array([1,1])
		for i in range(1, qubits):
			init = np.kron(init, minus)
		for i, k in enumerate(init):
			if i % 2 == 1:
				init[i] = -k
	else:    
		init = np.zeros(2**qubits)
		init[2**qubits - 1] = 1
        	
	return init
	
def break_pb(n_qubits, jx, jy, jz, periodic=True):
    '''
    Function that creates a matrix (Hamiltonian) that contains only the symmetry 
    breaking term.
    
    Args: 
        n_qubits (int): number of qubits that the Hamiltonian acts on.
        jx (float): Strength in X-direction.
        jy (float): Strength in Y-direction.
        jz (float): Strength in Z-direction.
    '''
    hamiltonian = QubitOperator()
    bpb = QubitOperator()
    if periodic:
        for i in range(n_qubits):
            hamiltonian += (
                QubitOperator(f'X{i} X{(i+1)%n_qubits}', jx) +
                QubitOperator(f'Y{i} Y{(i+1)%n_qubits}', jy) +
                QubitOperator(f'Z{i} Z{(i+1)%n_qubits}', jz))
        bpb += (
            QubitOperator(f'X{i} X{(0)%n_qubits}', jx) +
            QubitOperator(f'Y{i} Y{(0)%n_qubits}', jy) +
            QubitOperator(f'Z{i} Z{(0)%n_qubits}', jz)) 
     
    return bpb
	
