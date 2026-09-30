
import numpy as np
def orthogonal_projection(v, L):


	"""
	Compute the orthogonal projection of vector v onto line L.

	:param v: The vector to be projected
	:param L: The line vector defining the direction of projection
	:return: List representing the projection of v onto L
	"""
	a=np.array(v)
	b=np.array(L)
	return ((a @ b)/(b @ b)) * b
