import numpy as np
def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
	if mode == 'column':
		matrix=np.transpose(matrix)
		res=[np.mean(i) for i in matrix ]
	elif mode =='row':
		res=[np.mean(i) for i in matrix ]
	else:
		raise ValueError("error")


	return res