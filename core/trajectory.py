from typing import NamedTuple

import numpy as np
import scipy.stats as stats

from core.environnement import Environment
from core.niw import MultivariateParams

class ObservedTrajectory(NamedTuple):
    observations : np.ndarray
    actions : np.ndarray

    def check_valid(self):
        return len(self.actions) == len(self.observations)

class CompleteTrajectory(NamedTuple):
    states : np.ndarray
    actions : np.ndarray
    observations : np.ndarray

    def check_valid(self):
        return len(self.actions) == len(self.observations) == len(self.states) - 1
