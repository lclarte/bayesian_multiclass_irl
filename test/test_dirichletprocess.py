import unittest
import sys
sys.path.append("..")

import numpy as np

import core.dirichletprocess as dirichletprocess

class TestDirichletProcess(unittest.TestCase):
    def setUp(self):
        self.dp = dirichletprocess.DirichletProcess(tau = 1.)

    def tearDown(self):
        pass

    def test_sample_class(self):
        """
        Test la fonction sample_class, verifie qu'on obtient bien un entier positif 
        Ne verifie pas que la distribution est celle attendue (un peu complique a verifier)
        """
        try:
            sample = self.dp.sample_class()
            self.assertTrue(type(sample) == int and sample >= 0)
        except Exception as e:
            print("Sampling from DP failed")
            print(e)
            self.assertFalse(True)

    def test_sample_classes(self):
        try:
            size = (5, 2)
            samples = self.dp.sample_classes(size)
            self.assertTrue(samples.dtype == int and np.all(samples >= 0) and samples.shape == size )
        except Exception as e:
            print("Sampling several classes from DP failed : ")
            print(e)
            self.assertFalse(True)

if __name__ == '__main__':
    unittest.main()