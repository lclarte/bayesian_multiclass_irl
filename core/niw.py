from typing import NamedTuple

import numpy as np
import scipy.stats as stats

class NIWParams(NamedTuple):
    mu_mean : np.array
    mu_scale : float
    # Contrairement a ce que son nom indique, Sigma_mean n'est pas l'esperance de la matrice de covariance
    # En effet, l'esperance vaut Sigma_mean / (Sigma_scale - p - 1) avec p la dimension de notre espace 
    Sigma_mean : np.ndarray
    Sigma_scale : float

    def __repr__(self):
        return str(id(self)) + ":" + str(self.mu_mean) + '/' + str(self.mu_scale) + '/' + str(self.Sigma_mean) + '/' + str(self.Sigma_scale)

class MultivariateParams(NamedTuple):
    mu : np.array
    Sigma : np.array

    def check_valid(self):
        return len(self.mu) == self.Sigma.shape[0] == self.Sigma.shape[1]

def default_niw_prior(n, scale = 1.0) -> NIWParams:
    mu_mean = np.zeros(n)
    mu_scale = 1.
    Sigma_mean = scale * np.eye(n)
    # pour avoir une moyenne, il faut Sigma_scale > n + 1
    Sigma_scale = float(n+2)
    return NIWParams(mu_mean = mu_mean, mu_scale = mu_scale, Sigma_mean = Sigma_mean, Sigma_scale = Sigma_scale)

def sample_niw(params : NIWParams, size) -> (np.ndarray, np.ndarray):
    """
    returns : 
        - mu : ndarray de taille (size, n)
        - Sigma : ndarray de taille (size, n, n)
    """
    mu_0, k_0, Sigma_0, nu_0 = params.mu_mean, params.mu_scale, params.Sigma_mean, params.Sigma_scale

    Sigma = stats.invwishart.rvs(df = nu_0, scale = Sigma_0, size=size)
    # reshape if required (le comportement est pas certain pour len(size) > 1)
    if len(size) > 1:
        Sigma = np.reshape(Sigma, size + Sigma_0.shape)
    mu = np.zeros(size + mu_0.shape)

    # iterate over all the covariance matrices
    it = np.nditer(np.zeros(size), flags=['multi_index'])
    for _ in it:
        indx = it.multi_index
        mu[indx] = stats.multivariate_normal.rvs(mean = mu_0, cov = Sigma[indx] / k_0)
    
    return mu, Sigma

def niw_posterior_params(prior : NIWParams, samples : np.ndarray) -> NIWParams:
    """
    Gives the posterior parameters of NIW given observations of the w_1, ..., w_N sampled from N(mu, Sigma)
    and prior mu_0, k_0, Sigma_0, nu_0
    """

    mu_0, k_0, Sigma_0, nu_0 = prior.mu_mean, prior.mu_scale, prior.Sigma_mean, prior.Sigma_scale

    k, n = samples.shape
    w_mean = np.mean(samples, axis=0)
    w_cov  = (samples - w_mean).T @ (samples - w_mean)
    
    nu_post = nu_0 + k
    k_post   = k_0  + k

    mu_post = (k_0 * mu_0 + k * w_mean) / k_post
    w_tilde = np.reshape(w_mean - mu_0, newshape=(1, n))
    Sigma_post = Sigma_0 + w_cov + (k_0 * k) / k_post * (w_tilde.T @ w_tilde)

    return NIWParams(mu_post, k_post, Sigma_post, nu_post)
