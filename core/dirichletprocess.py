# dirichletprocess.py
# contains function to sample from dirichlet process with the stick breaking process 
# cf. https://en.wikipedia.org/wiki/Dirichlet_process#The_stick-breaking_process

import numpy as np
import scipy.stats as stats

from typing import NamedTuple

def sample_beta(size, tau):
    """
    Sample from beta distribution Beta(1.0, tau)
    parameters:
        - size : tuple 
        - tau : real positive number
    """
    return stats.beta.rvs(1., tau, size=size)

def compute_class_probabilities(betas):
    """
    Computes a (non normalized) list of probas using the list of betas : 
    p_k = beta_k \prod_{i = 1}^{k - 1} (1 - beta_i) 
    """
    # note : if the list of betas were infinite, the ps would sum to 1
    ps    = np.zeros((len(betas),))
    ps[:]    = betas[:]
    ps[1:] *= np.cumprod(1 - betas[:-1])  
    return ps

class DirichletProcess:
    """
    Allows to sample exactly from the dirichlet process i.e with an infinite number of classes. 
    """

    tau     : float

    def __init__(self, tau):
        # nombre de classe qu'on sample initialement
        self.tau = tau
        self.num_betas = 50
        self.betas = np.zeros((0))
        self.probas = np.zeros((0))
        self.probas_sum = []
        
    def add_betas_probas(self):
        betas = sample_beta(self.num_betas, self.tau)
        probas = compute_class_probabilities(betas)
        
        self.betas = np.concatenate((self.betas, betas))
        self.probas = np.concatenate((self.probas, probas))
        self.probas_sum.append(np.sum(probas))
        if not (1 >= self.probas_sum[-1] >= 0.):
            raise Exception("Probas in block do not sum to 1 for sampling from stick-breaking DP")

    def sample_class_from_probas(self, block):
        """
        Echantillonne une classe comprise entre block * K et (block + 1) * K - 1 (les classes commencent a 0)
        """
        start = block * self.num_betas
        normalized_probas = self.probas[start:start+ self.num_betas] / np.sum(self.probas[start:start+ self.num_betas])
        return start + np.random.choice(self.num_betas, p=normalized_probas)

    def sample_class(self):
        """
        Sample a single class from the dirichlet process 
        """
        block = 0
        while True:
            # Ajouter des classes si le bloc courant est au dela du nombre de blocs
            while block >= len(self.probas_sum):
                self.add_betas_probas()
            # on teste si on va tirer la classe du bloc courant 
            sample_from_current_block = np.random.binomial(1, p=self.probas_sum[block])
            if sample_from_current_block == 1:
                return self.sample_class_from_probas(block)
            else:
                block += 1

    def sample_classes(self, size):
        """
        Sample classes given by the tuple size
        """
        classes = np.zeros(shape=size, dtype=int)
        it = np.nditer(classes, flags=['multi_index'])
        for indx in it:
            indx = it.multi_index
            classes[indx] = self.sample_class()
        return classes