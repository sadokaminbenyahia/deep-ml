import numpy as np

def gaussian_elimination(A, b):
	"""
	Solves the system Ax = b using Gaussian Elimination with partial pivoting.
    
	:param A: Coefficient matrix
	:param b: Right-hand side vector
	:return: Solution vector x
	"""	
	m,n=A.shape
	if (np.linalg.matrix_rank(A)==n):
		return np.dot(np.linalg.inv(A),b)
	return np.zeros_like(b)
