from typing import NamedTuple

import numpy as np
import scipy.stats as stats

class NIWParams(NamedTuple):
    mu_mean : np.array
    mu_scale : float
    # TODO : Contrairement a ce que son nom indique, Sigma_mean n'est pas l'esperance de la matrice de covariance
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

def default_niw_prior(n) -> NIWParams:
    mu_mean = np.zeros(n)
    mu_scale = 1.
    Sigma_mean = np.eye(n)
    Sigma_scale = float(n)
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
    Donne la loi posterieure de mu, Sigma a partir de l'observations des w_1, ..., w_N 
    et du prior mu_0, k_0, Sigma_0, nu_0
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

def norminvwishart_pdf(mu : np.ndarray, Sigma : np.ndarray, params : NIWParams) -> float:

    try:
        mu_likelihood = stats.multivariate_normal.pdf(x = mu, mean = params.mu_mean, cov = Sigma / params.mu_scale)
        Sigma_likelihood = stats.invwishart.pdf(x = Sigma, df = params.Sigma_scale, scale = params.Sigma_mean)
        return mu_likelihood * Sigma_likelihood
    
    except Exception as e:
        M = Sigma / params.mu_scale
        Minv = np.linalg.inv(M)
        print('Error in pdf of normal inverse wishart :', np.linalg.det(M), ' & ', np.linalg.det(Minv))
        
        raise Exception()

def monte_carlo_niw_likelihood(w : np.ndarray, params : NIWParams, M = 10, mus_log : np.ndarray = None, Sigmas_log : np.ndarray = None) -> float:
    """
    Ne semble pas fonctionner pour l'instant 
    """
    assert mus_log is None or len(mus_log) == M
    assert Sigmas_log is None or len(Sigmas_log) == M
    likelihood = 0.

    mus, Sigmas = sample_niw(params, size=(M, ))
    if not (mus_log is None):
        mus_log[:] = mus
    if not (Sigmas_log is None):
        Sigmas_log[:] = Sigmas
    
    den = 0

    for m in range(M):

        try:
            niw_likelihood = norminvwishart_pdf(mus[m], Sigmas[m], params)
            likelihood += stats.multivariate_normal.pdf(w, mean=mus[m], cov=Sigmas[m]) * niw_likelihood
            den += 1
        except Exception as e:
            pass

    return (likelihood / float(den))