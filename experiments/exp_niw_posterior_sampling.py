import sys
sys.path.append("..")

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

import core.niw as niw
import core.environnement as env
import core.dirichletprocess as dp 
import core.policy as policy
import core.inference as inference

parameters = {
'n' :  2,

'mu_0' : np.zeros(2),
'k_0'  : 1.0,
'Sigma_0' : np.identity(2),
'nu_0' : 2,

'k_max' : 10000,
}


def test_niw_posterior_sampling(p):
    """"
    Teste la loi posterieure de Normale Inverse Wishart : on sample (mu, Sigma) du prior puis on sample des ws de la gaussienne multivariee N(mu, Sigma).
    """
    mu_0 = p['mu_0']
    Sigma_0 = p['Sigma_0']
    k_0 = p['k_0']
    nu_0 = p['nu_0']

    k_max = p['k_max']

    prior = niw.NIWParams(mu_0, k_0, Sigma_0, nu_0)

    mu, Sigma = niw.sample_niw(prior, size=(1,))
    print('Sampled mean : ', mu[0])
    print('Covariance matrix : ', Sigma[0])

    ws = stats.multivariate_normal.rvs(mean=mu[0], cov=Sigma[0], size=(k_max,))
    # distances entre le vrai mu et le MAP de la loi posterieure 
    distances = [0]*k_max

    # compute posterior law with partial observation of the samples 
    for k in range(k_max):
        mu_p, k_p, Sigma_p, nu_p = niw.niw_posterior_params(prior, ws[:k])
        distances[k] = np.linalg.norm(x = mu_p - mu)

    plt.plot(distances) 
    plt.yscale("log")
    plt.title('Distance entre $\mu$ infere et le vrai')
    plt.show()

if __name__ == "__main__":
    test_niw_posterior_sampling(parameters)
