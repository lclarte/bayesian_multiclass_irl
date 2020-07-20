import unittest
import sys
sys.path.append("..")

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
        except:
            print("Sampling from DP failed")
            self.assertFalse(True)

if __name__ == '__main__':
    unittest.main()