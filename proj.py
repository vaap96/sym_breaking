import scipy.linalg as la
import numpy as np

def projectors(group, qubits):
	
	'''
	Function to get the projectors for different group representations. Returns
	dictionary with keys the eigenvalues of the group generator and values np 
	arrays that correspond to the projectors of each eigenvalue. 
	
	Args:
		group (array): the generator of group representation for which we want to find projectors.
		qubits (int): the number of qubits that the Hamiltonian acts on. 
	'''

	eigvals, eigvecs = la.eig(group)

	eigvals_r = eigvals.round(6)
	eigvecs_r = eigvecs

	vecs = []
	for i in range(len(eigvecs_r)):
		vecs.append(eigvecs_r[:, i]) 
    
	dict_c = {}
	for l in eigvals_r:
		if l not in dict_c:
			dict_c[l] = np.zeros((2 ** qubits, 2 ** qubits), dtype=np.complex_)
    
	for i, vec in enumerate(vecs):
		for k in dict_c.keys():
			if k == eigvals_r[i]:
				dict_c[eigvals_r[i]] += np.outer(vec, vec.conj())

	return dict_c
