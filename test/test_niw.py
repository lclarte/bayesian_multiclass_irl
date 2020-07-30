import unittest
import sys
sys.path.append('..')

import numpy as np

from irl.core import niw

class TestNIW(unittest.TestCase):
    def setUp(self):
        self.n = 2
        self.params = niw.NIWParams(mu_mean = np.zeros(shape=(self.n,)),
                                    mu_scale = 1.0,
                                    Sigma_mean = np.eye(self.n),
                                    Sigma_scale = 2.0)

    def tearDown(self):
        pass


    def test_sample_niw_size(self):
        """
        Verifie que le sampling de NIW renvoie des mu, Sigma de la taille desiree
        """
        size = (1, 2, 3)
        mu, Sigma = niw.sample_niw(self.params, size)
        self.assertTrue(mu.shape == size + (self.n, ) and Sigma.shape == size + (self.n, self.n))

    def test_posterior_niw(self):
        """
        TODO : Teste avec un exemple simple que le posterior de normal inverse wishart marche bien
        """
        pass

    def test_norminvwishart_pdf(self):
        """
        TODO : Teste que la likelihood de la loi normale inverse wishart marche bien 
        """
        pass

if __name__ == '__main__':
    unittest.main()