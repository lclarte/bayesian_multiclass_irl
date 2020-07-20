import numpy as np
import scipy.stats as stats

from typing import NamedTuple

def sample_beta(size, tau):
    """
    Sample from beta distribution 
    """
    return stats.beta.rvs(1., tau, size=size)

def compute_class_probabilities(betas):
    ps    = np.zeros((len(betas),))
    ps[:]    = betas[:]
    ps[1:] *= np.cumprod(1 - betas[:-1])  
    return ps

class DirichletProcess:
    """
    Classe contenant les parametres tau  & (mu_0, Sigma_0, k_0, nu_0) pour le processus de Dirichlet qui genere les params.
    mu, SIgma
    Remarque : nu_0 doit etre strictement plus grand que D - 1, avec D la tailel de Sigma_0
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

    def sample_class_from_probas(self, block):
        """
        Echantillonne une classe comprise entre block * K et (block + 1) * K - 1 (les classes commencent a 0)
        """
        start = block * self.num_betas
        normalized_probas = self.probas[start:start+ self.num_betas] / np.sum(self.probas[start:start+ self.num_betas])
        return start + np.random.choice(self.num_betas, p=normalized_probas)

    def sample_class(self):
        """
        Attention : il faut un nombre infini de betas pour que ca somme a 1. 
        On sample une var Bernoulli(Somme(self.probas)). Si elle vaut 1, alors on fait np.choice(self.probas / Somme(probas))
        Sinon, on resample des nouvelles probas et on recommence    
        """
        block = 0
        while True:
            while block >= len(self.probas_sum):
                self.add_betas_probas()
            sample_from_current_block = np.random.binomial(1, p=self.probas_sum[block])
            if sample_from_current_block == 1:
                return self.sample_class_from_probas(block)
            else:
                block += 1

    def sample_classes(self, size):
        """
        Retourne un np array contenant des classes
        """
        classes = np.zeros(shape=size, dtype=int)
        it = np.nditer(classes, flags=['multi_index'])
        for indx in it:
            indx = it.multi_index
            classes[indx] = self.sample_class()
        return classes