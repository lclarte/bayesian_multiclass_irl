import unittest
import sys
sys.path.append("..")

import numpy as np
from scipy.special import gamma

import core.niw as niw
import core.gibbs_class as gibbs_class

class TestGibbsClass(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_posterior_prior_ratio(self):
        """
        Teste que la fonction posterior_prior_ration renvoie le bon resultat (sur un exemple simple)
        """
        # 1) On essaie dans le cas ou prior = posterior
        prior = niw.default_niw_prior(2)
        nu_0 = 3.0
        posterior = niw.NIWParams(mu_mean = prior.mu_mean, mu_scale = prior.mu_scale, Sigma_mean = prior.Sigma_mean, Sigma_scale = nu_0)

        ratio = gibbs_class.posterior_prior_ratio(prior, posterior)
        true_value = (1 / np.pi) * gamma(nu_0 / 2. ) / gamma((nu_0 - 2.) / 2.)

        bool_same = (true_value == ratio)

        # TODO : faire un autre cas simple

        self.assertTrue(bool_same)


    def test_posterior_class_integral(self):
        pass

    def test_posterior_class_gibbs_sampling(self):
        pass