import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

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

class DirichletProcess():
    """
    Classe contenant les parametres tau  & (mu_0, Sigma_0, k_0, nu_0) pour le processus de Dirichlet qui genere les params.
    mu, SIgma
    Remarque : nu_0 doit etre strictement plus grand que D - 1, avec D la tailel de Sigma_0
    """

    tau     : float

    def __init__(self):
        # nombre de classe qu'on sample initialement
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

class Environment(NamedTuple):
    """
    Classe contenant les probas de transition et d'observation et les features pour la reward 
    Attention a ce que les matrices soient by normalisees
    Taille des matrices : 
        - trans_matx : S x A x S
        - obsvn_matx : S x A x O
        - features   : S x A
        - init_dist  : S
    """
    trans_matx : np.ndarray
    obsvn_matx : np.ndarray
    features   : np.ndarray
    init_dist  : np.ndarray
    gamma      : float = 0.9

    def check_compatible_sizes(self):
        S1, A1, S2 = self.trans_matx.shape
        S3, A2, O  = self.obsvn_matx.shape
        S4, A3     = self.features.shape
        S5         = self.init_dist.shape

        return (S1 == S2 == S3 == S4 == S5) and (A1 == A2 == A3)

def Trajectory(NamedTuple):
    """
    Contient la liste des etats, des observations et des actions prises dans une trajectoire
    """
    actions      : np.ndarray = None
    observations : np.ndarray = None
    states       : np.ndarray = None

    def check_compatible_sizes(self):
        return (len(self.actions) == len(self.observations) == len(self.states) - 1)

    def get_chain_potentials(self, policy : np.ndarray, env : Environment):
        """
        retourne les potentiels binaires et unaires d'une chaine ou les variables a inférer sont les états
        et les variables observées sont les observations et les actions.
        Ces potentiels sont conditionnés par la police et les matrices de transition / observation 
        """
        assert (not (self.actions is None)) and (not (self.observations is None))
        T = len(observations) + 1
        S, A, _ = env.trans_matx.shape
        # binary[t] : entre s_t et s_{t+1}
        unary, binary = np.zeros(shape=(T+1, S)), np.zeros(shape=(T, S))
        # first state is special case
        unary[0, :] = policy[:, self.actions[0]] * env.init_dist[:]
        for t in range(1, T+1):
            unary[t, :] = policy[:, self.actions[t]] * env.obsvn_matx[:, self.actions[t-1], self.observation[t]]
            binary[t-1] = env.trans_matx[:, self.actions[t-1], :]
        return unary, binary