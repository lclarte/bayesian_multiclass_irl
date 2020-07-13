import unittest
import sys
sys.path.append("..")

import numpy as np

import core.policy as policy

class TestPolicy(unittest.TestCase):
    def test_q_function_accuracy(self):
        """
        Test que le calcule de la fonction q* renvoie le bon resultat 
        """
        reward_function = np.array([
            [1., 0.], 
            [10., 10.]
        ])
        transition = np.zeros(shape=(2, 2, 2))
        transition[0] = np.eye(2)
        transition[1] = np.array([[0., 1.], [0., 1.]])

        q_star = policy.q_function(reward_function, transition, gamma=0.9)
        true_q_star = np.array([[82., 90.], [100., 100.]])
        
        self.assertTrue(np.allclose(q_star, true_q_star, atol=1e-1))

    def test_q_function_not_NaN(self):
        """
        Teste qu'il n'y a pas de cas ou la fonction q* renvoie Nan
        """
        pass

if __name__ == "__main__":
    unittest.main()