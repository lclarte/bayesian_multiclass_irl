import unittest
import sys
sys.path.append('..')

import numpy as np

from core.metrics import *
from core.environnement import *

class TestMetrics(unittest.TestCase):
    def setUp(self):
        S, A, O = 5, 2, 5
        n = 2
        self.transition = random_transition_matrix(S, A)
        self.features = np.random.rand(S, A, n)

    def tearDown(self):
        pass

    def test_quadratic_q(self):
        ws, ws2 = np.random.rand(2), np.random.rand(2)
        try:
            quadratic_q(ws, ws2, self.features, self.transition, 0.9)
        except:
            self.assertTrue(False)
        