
import numpy as np

def r_squared(y_true, y_pred):
	# Write your code here
	SSR=np.sum((y_true - y_pred)**2)
	mean=[np.mean(y_true)]*len(y_true)
	mean=np.array(mean)
	SST=np.sum((y_true - mean)**2)
	return (1-(SSR/SST))
	
	pass
