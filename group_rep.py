import numpy as np
import sys
import scipy.linalg as la

def cyclic(n_qubits):

	'''
	Function for the representation of the cyclic group for a specific number of
	qubits. 
	Returns a list with all the elements of the cyclic group 
	representation.

	Args: 
		n_qubits (int): number of qubits that the Hamiltonian acts on. 
	'''
	
	c1 = np.zeros((2 ** n_qubits, 2 ** n_qubits))

	# Create initial vectors with all combinations.
	l_qubits = []
	for k in range(2 ** n_qubits):
		l_qubits.append([int(i) for i in list(format(k, str(0) + str(n_qubits) + 'b'))])
	
	# Make first cyclic change.
	for lis in l_qubits:
		lis.append(lis.pop(0))

    # Trasnform to integers in order to find the correct position of ones.
	int_bit = []
	for l in l_qubits:
		int_bit.append(int("".join(str(i) for i in l),2))

    # Place the ones in the appropriate positions.
	for i in range(2 ** n_qubits):
		c1[:,i][int_bit[i]] = 1

    # Create the group.
	c_group = []
	for i in range(1, n_qubits+1):
		c_group.append(np.linalg.matrix_power(c1, i))
	
	return c_group

def reflection(n_qubits):
	
	'''
	Function for the representation of the reflection group for a specific 
	number of qubits. Returns a list with all the elements of the reflection 
	group representation.

	Args: 
		n_qubits (int): number of qubits that the Hamiltonian acts on. 
	'''	
    
	if (n_qubits % 2) == 1: 
        # Initialize r1.
		r1 = np.zeros((2 ** n_qubits, 2 ** n_qubits))

        # Create initial vectors with all combinations.
		l_qubits = []
		for k in range(2 ** n_qubits):
			l_qubits.append([int(i) for i in list(format(k, str(0) + str(n_qubits) + 'b'))])

        # Make reflection change.
		l_qub = []
		if int(n_qubits/2) < 2:
			for l in range(len(l_qubits)):
				l_qubits[l][0],  l_qubits[l][-1] = l_qubits[l][-1], l_qubits[l][0]
				l_qub = l_qubits

		elif int(n_qubits/2) > 1:
			for l in range(len(l_qubits)):
				b = l_qubits[l][int(n_qubits/2)+1:]
				for i in range(int(len(b)/2)):
					b[i], b[-(i+1)] = b[-(i+1)], b[i]
				b.append(l_qubits[l][int(n_qubits/2)])
				e = l_qubits[l][:int(n_qubits/2)]
				for i in range(int(len(e)/2)):
					e[i], e[-(i+1)] = e[-(i+1)], e[i]
				l_qub.append(b+e)

        # Trasnform to integers in order to find the correct position of ones.
		int_bit = []
		for l in l_qub:
			int_bit.append(int("".join(str(i) for i in l),2))

        # Place the ones in the appropriate positions.
		for i in range(2 ** n_qubits):
			r1[:,i][int_bit[i]] = 1

	elif (n_qubits % 2) == 0:
        # Initialize r1.
		r1 = np.zeros((2 ** n_qubits, 2 ** n_qubits))
        
        # Create initial vectors with all combinations.
		l_qubits = [i for i in range(n_qubits ** 2)]
		l_qubits = []
		for k in range(2 ** n_qubits):
			l_qubits.append([int(i) for i in list(format(k, str(0) + str(n_qubits) + 'b'))])
        
        # Make reflection change.
		l_qub = []
		if int(n_qubits/2) < 2:
			for l in range(len(l_qubits)):
				l_qubits[l][0],  l_qubits[l][-1] = l_qubits[l][-1], l_qubits[l][0]
				l_qub = l_qubits

		elif int(n_qubits/2) > 1:
			for l in range(len(l_qubits)):
				b = l_qubits[l][int(n_qubits/2):]
				for i in range(int(len(b)/2)):
					b[i], b[-(i+1)] = b[-(i+1)], b[i]
				e = l_qubits[l][:int(n_qubits/2)]
				for i in range(int(len(e)/2)):
					e[i], e[-(i+1)] = e[-(i+1)], e[i]
				l_qub.append(b+e)

        # Trasnform to integers in order to find the correct position of ones.
		int_bit = []
		for l in l_qub:
			int_bit.append(int("".join(str(i) for i in l),2))
        
        # Place the ones in the appropriate positions.
		for i in range(2 ** n_qubits):
			r1[:,i][int_bit[i]] = 1
                
	r_group = [r1, r1@r1]

	return r_group

def group_ham(ham, group):
	
	'''
	Change the basis of the Hamiltonian using a specific group representation.
	Returns the result after adding all the change of basis matrices using each
	element of the group representation and dividing by the number of elements 
	that the group representation contains.
	
	Args:
		ham (np array): the Hamiltonian that we want to diagonalise.
		group (list): group that we will use for the change of basis. 
	'''
	
	# Get the sum of the change of basis from each group element.
	hh_list = []
	for gr in group:
		hh_list.append(gr.transpose() @ ham @ gr)
    
	# Divide the sum by the number of elements that the group contains.
	hh = sum(hh_list) / len(group)
	
	return hh

def cr_block(group):
	
	'''
	Function that creates the block structure of a given Hamiltonian using a 
	specific group representation. The block structure only depends on the 
	group that we will use. The function returns an ordered version of the 
	eigenvectors of the generator of the group.

	Args:
		group (np array): group generator that we will use for the change of basis.  
	'''

    # Find the eigendecomposition of group.
	eigvals, eigvecs = la.eig(group[0])
    
    # Round eigvals.
	eigvals_r = eigvals.round(8)

    # Find unique values in the eigvals.
	unq = []
	for i in eigvals_r:
		if i not in unq:
			unq.append(i)
     
    # Create dictionary with the indexes of identical eigvals.
	dict_val = {}
	for i in unq:
		dict_val[i] = []
     
	for i,k in enumerate(eigvals_r):
		for l in dict_val.keys():
			if k == l:
				dict_val[k].append(i)
    
    # Sort values of dictionary based on length (descending order).
	counts = list(dict_val.values())
	counts.sort(key = len, reverse=True)
    
    # Create list with the appropriate positions in order to get the block structure.
	entire = []
	for i in counts:
		entire.extend(i)
    
    # Implement the change of order.
	eigvecs_or = eigvecs[:,entire]

	return eigvecs_or
