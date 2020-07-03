import numpy as np
import scipy.stats as stats

from typing import NamedTuple

class DirichletProcessParams():
    """
    Classe contenant les parametres tau  & (mu_0, Sigma_0, k_0, nu_0) pour le processus de Dirichlet qui genere les params.
    mu, SIgma
    Remarque : nu_0 doit etre strictement plus grand que D - 1, avec D la tailel de Sigma_0
    """

    Sigma_0 : np.ndarray
    mu_0    : np.ndarray
    k_0     : float
    nu_0    : float

    tau     : float

    def sample_norm_inv_wish(self, size):
        """
        Retourne un tuple (mu, Sigma) tires de la loi Normale x inverse Wishart. 
        parameters:
            - size : tuple (d0, ..., dn)
        returns : 
            - mu : np.array de taille (d0, ..., dn, N)
            - Sigma : np.array de taille (d0, ..., dn, N, N)
        """
        Sigma = stats.invwishart.rvs(self.nu_0, self.Sigma_0, size)
        mu = np.zeros(size + self.mu_0.shape)
        # iterate over all the covariance matrices 
        it = np.nditer(np.zeros(size), flags=['multi_index'])
        for _ in it:
            indx = it.multi_index
            mu[indx] = stats.multivariate_normal.rvs(mean = self.mu_0, cov = Sigma[indx] / self.k_0)
        return mu, Sigma

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