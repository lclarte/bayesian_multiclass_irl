# empomdp.py
# class implementing the EM-POMDP algorithms

from typing import List

import numpy as np
from scipy import stats, optimize, special
from sklearn import mixture

from core.environnement import *
from core.trajectory import *
from core.inference import *


class EMPOMDP:
    def __init__(self, *, verbose = False, n_iter = 10, n_classes = 1):
        self.verbose = verbose
        self.n_iter = n_iter
        self.n_classes = n_classes

        self.inf_mus = None
        self.inf_Sigmas = None
        self.inf_classes = None
        self.inf_ws = None
        
    def infer(self, env : Environment, trajectories : List[ObservedTrajectory], eta : float):
        """
        Non hierarchical bayesian version of the algorithm : components of the gaussian mixture are identified with EM
        parameters : 
            - env : environment in which all the MDPs evolve
            - trajectories : observed trajectories
        """
        _, _, n = env.features.shape
        M = len(trajectories)
        
        # mixture to model the distribution of weights = gaussian mixture 
        # no NIW prior here 
        gaussianmixture = mixture.GaussianMixture(n_components=self.n_classes)

        infered_ws = np.zeros(shape=(M, n))
        infered_classes = np.zeros(shape=(M, ), dtype=int)
        infered_mus = np.zeros(shape=(self.n_classes, n))
        infered_Sigmas = np.array([np.eye(n) for _ in range(self.n_classes)])

        for k in range(self.n_iter):
            if self.verbose:
                print('Iteration #' + str(k))
                print('infered_mus = ' + str(infered_mus))
                print('infered_Sigmas = ' + str(infered_Sigmas))

            # step 1 : compute all ws 
            for m in range(M):
                c = infered_classes[m]
                params = MultivariateParams(mu = infered_mus[c], Sigma = infered_Sigmas[c])  
                infered_ws[m] = map_w_from_observed_trajectory(trajectories[m], params, eta, env)

            # step 2 : update all class assignements + parameters
            gaussianmixture.fit(infered_ws)
            infered_mus = gaussianmixture.means_
            infered_Sigmas = gaussianmixture.covariances_

            infered_classes = list(map(lambda x : get_class(x, infered_mus, infered_Sigmas), infered_ws))

        self.inf_mus     = infered_mus
        self.inf_Sigmas  = infered_Sigmas
        self.inf_classes = infered_classes
        self.inf_ws      = infered_ws