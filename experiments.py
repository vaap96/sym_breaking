import numpy as np
import ham_rep
import group_rep
import openfermion as of
import sys
import scipy.linalg as la
from qiskit import quantum_info as qi 
from proj import projectors

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=np.inf)

qubits = 4
jx = -0.5
jy = -1
jz = -1

ham_pb = ham_rep.heisenberg_hamiltonian(qubits, jx, jy, jz)
ham_pb = of.get_sparse_operator(ham_pb).toarray()
ham_ob = ham_rep.heisenberg_hamiltonian(qubits, jx, jy, jz, periodic = False)
ham_ob = of.get_sparse_operator(ham_ob).toarray()

eigvals, eigvecs = la.eig(ham_pb)
gs = np.array(eigvecs[:,2])

cyclic = group_rep.cyclic(qubits)[0]
proj_c = projectors(cyclic, qubits)

commutator_pb = (ham_pb @ cyclic) - (cyclic @ ham_pb)
commutator_ob = (ham_ob @ cyclic) - (cyclic @ ham_ob)

pauli_x = np.array([[0,1],[1,0]])

def break_degen(n_qubits):
    hamiltonian = of.QubitOperator()
    for i in range(n_qubits):
        hamiltonian += of.QubitOperator(f'X{i}')

    return hamiltonian

bs = of.get_sparse_operator(break_degen(qubits)).toarray()
ham_bs = ham_pb + bs

commutator_bs = (ham_pb @ bs) - (cyclic @ bs)
