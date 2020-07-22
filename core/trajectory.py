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

def get_chain_potentials(traj : ObservedTrajectory, policy : np.ndarray, env : Environment):
        """
        Retourne les potentiels binaires et unaires d'une chaine ou les variables a inférer sont les états
        et les variables observées sont les observations et les actions.
        Ces potentiels sont conditionnés par la police et les matrices de transition / observation 
        Permet de calculer la proba conditionnelle P(s_1, ..., s_T | a_1, ..., a_T, o_1, ..., o_T)
        NORMALEMENT, NE PERMET PAS DE CALCULER LE MAP DE W via sa normalisation
        """
        assert len(traj.actions) == len(traj.observations), str(len(traj.actions)) + " " + str(len(traj.observations))

        T = len(traj.observations)
        S, A, _ = env.trans_matx.shape
        # binary[t] : entre s_t et s_{t+1}
        unary, binary = np.zeros(shape=(T+1, S)), np.zeros(shape=(T, S, S))
        # first state is special case
        unary[0, :] = policy[:, traj.actions[0]] * env.init_dist
        for t in range(1, T+1):
            if t == T:
                # pas de decision prise a ce moment la (pas de policy[ ... ])
                unary[T] = env.obsvn_matx[:, traj.actions[T-1], traj.observations[T-1]] 
            else: # t <= T - 1
                unary[t] = policy[:, traj.actions[t]] * env.obsvn_matx[:, traj.actions[t-1], traj.observations[t-1]]
            binary[t-1] = env.trans_matx[:, traj.actions[t-1], :]
        return unary, binary
