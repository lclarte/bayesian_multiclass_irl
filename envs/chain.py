import sys
sys.path.append("..")

import numpy as np

import core.environnement as environnement

def get_chain_env(S : int, eps=.1) -> environnement.Environment:
    """
    Dans cet environnement, on a que deux etats interessants : le dernier ou l'avant dernier
    arguments:
        - S   : nombre d'etats 
        - eps : proba de retourner a l'etat initial 0 
    """
    assert S > 2
    # two actions : go right or stay in same 
    A, O = 2, S
    obsvn_matx = environnement.noisy_id_observation_matrix(S, A, O, eps=0.)
    
    # transition : avec une certaine proba, on retourne au premier etat
    trans_matx = np.zeros(shape=(S, A, S))
    # action 0 : stay here, action 1 : go right
    for s in range(S-1):
        trans_matx[s, 0, s+1] += 1 - eps
        trans_matx[s, 0, 0]   += eps
        
        trans_matx[s, 1, s]   += 1 - eps
        trans_matx[s, 1, 0]   += eps
        
    trans_matx[S-1, 0, S-1] = trans_matx[S-1, 1, S-1] = 1 - eps
    trans_matx[S-1, 0, 0]   = trans_matx[S-1, 1, 0]   = eps
    
    features = np.zeros(shape=(S, A, 2))
    for n in range( S - 1 ):
        features[n, 0] = features[n, 1]     = [1., 0.]

    features[S - 1, 0] = features[S - 1, 1] = [0., 1.]
    
    init_dist = np.array([1.] + [0.]*(S-1))
    return environnement.Environment(obsvn_matx = obsvn_matx, features = features, trans_matx = trans_matx, init_dist = init_dist)