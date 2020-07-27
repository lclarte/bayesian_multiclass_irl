# bp.py
# Contient les fonctions pour faire de la belief propagation

from scipy.special import logsumexp
import numpy as np

def compute_chain_normalization(log_psi_1 : np.ndarray, log_psi_2 : np.ndarray):
	"""
	Compute THE LOG OF the normalization constant given the potentials. Compute forward 
	and backward messages then compute the marginals and lastly partition
	function.

	arguments : 
		- log_psi_1 : log of potentials with 1 variable. Shape is (n, K)
					  log_psi_1[i, x] = Psi_i(x)
		- log_psi_2 : log of potentials for 2 variables. Shape is (n - 1, K, K)	
					  log_psi_2[i, x, y] = Psi_{i, i+1}(x, y)
	"""
	# TODO : Voir comment integrer ceci dans le code 
	# np.seterr(all='raise')
	try:
	
		n, K = log_psi_1.shape
		# forward_messages[i] = \mu_{i-1 -> i}
		forward_messages = np.zeros(shape=(n, K))
		# backward messages[i] = \mu_{i+1 -> i}
		backward_messages = np.zeros(shape=(n, K))
		for k in range(1, n):
			forward_messages[k] = logsumexp(forward_messages[k-1] + log_psi_1[k] + log_psi_2[k - 1].transpose(), axis=1)
		for k in reversed(range(n-1)):
			backward_messages[k] = logsumexp(backward_messages[k+1] + log_psi_1[k] + log_psi_2[k].transpose(), axis=1)
		# marginal distributions
		log_probas = forward_messages + backward_messages + log_psi_1
		# compute the log_Z using marginal at last node 
		return logsumexp(log_probas[-1])
	
	except Exception as e: 

		print("Error in chain normalization : ")
		print(e)
		return 0.

