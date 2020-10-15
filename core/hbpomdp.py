# hbpomdp.py
# Class implementing HierarchicalBayesian-POMDP algorithm

import numpy as np
from scipy import stats, optimize, special
from sklearn import mixture

from core.niw import *
from core.trajectory import *
from core.environnement import *
from core.inference import *

class HBPOMDP:
    def __init__(self, *, niw_prior : NIWParams = None, dp_tau : float = 1.,  verbose : bool = False, n_iter : int = 10):
        self.niw_prior = niw_prior
        self.dp_tau = dp_tau
        self.verbose = verbose
        self.n_iter = n_iter
        self.partially_observable_trajectories(True)

        self.inf_mus = None
        self.inf_Sigmas = None
        self.inf_classes = None
        self.inf_ws = None

    def partially_observable_trajectories(self, po : bool):
        if po:
            self.map_w_function = map_w_from_observed_trajectory 
        else:
            self.map_w_function = map_w_from_complete_trajectory 


    def infer(self, env : Environment, trajectories : List[ObservedTrajectory], eta : float):
        """
        Hierarchical Bayesian algorithm, makes use of the normal inv. wishart prior for the distribution of the weight vector. 
        NIW -> Multivariate Gaussian -> weight vector w 
        """
        M = len(trajectories)
        _, _, n = env.features.shape
        
        # because default niw_prior is None and it is an optional parameter
        if not self.niw_prior:
            self.niw_prior = default_niw_prior(n)
        
        means_prior = self.niw_prior.mu_mean
        precisions_prior = self.niw_prior.mu_scale
        covariances_prior = self.niw_prior.Sigma_mean
        # dof = degrees of freedom pour le sampling de la matrice de covariance
        dofs_prior = self.niw_prior.Sigma_scale
        gaussianmixture = mixture.BayesianGaussianMixture(weight_concentration_prior=self.dp_tau, mean_prior=means_prior, mean_precision_prior=precisions_prior,
                                                        covariance_prior=covariances_prior, degrees_of_freedom_prior=dofs_prior)

        infered_ws = np.zeros(shape=(M, n))
        infered_classes = np.zeros(shape=(M, ), dtype=int)

        # initialement, une seule classe et un seul parametre
        infered_mus = np.zeros(shape=(1, n))
        infered_Sigmas = np.array([covariances_prior])

        for k in range(self.n_iter):
            if self.verbose:
                print('Iteration ', k)
                print('infered_mus : ', infered_mus)
                print('infered_Sigmas : ', infered_Sigmas)

            # step 1 : compute all ws 
            for m in range(M):
                c = infered_classes[m]
                params = MultivariateParams(mu = infered_mus[c], Sigma = infered_Sigmas[c])  
                infered_ws[m] = self.map_w_function(trajectories[m], params, eta, env)
            
            # step 2 : update all class assignements + parameters
            gaussianmixture.fit(infered_ws)

            infered_mus = gaussianmixture.means_
            # infered_Sigmas will be the average value of the covariance on inverse wishart distribution : 
            # E(Sigma) = Sigma_0 / (nu - dim- 1)
            infered_Sigmas = gaussianmixture.covariances_
            # update class assignements 
            infered_classes = list(map(lambda x : get_class(x, infered_mus, infered_Sigmas), infered_ws))


        self.inf_mus     = infered_mus
        self.inf_Sigmas  = infered_Sigmas
        self.inf_classes = infered_classes
        self.inf_ws      = infered_ws