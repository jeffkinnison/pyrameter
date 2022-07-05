"""Spearmint-style gaussian process-based Bayesian optimization.

Functions
---------
bayes
    Spearmint-style gaussian process-based Bayesian optimization.
"""

import numpy as np
import scipy.stats
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF
from sklearn.preprocessing import StandardScaler

from pyrameter.methods.method import Method


class Bayesian(Method):
    """Spearmint-style gaussian process-based Bayesian optimization.

    Parameters
    ----------
    n_samples : int
        The number of candidate samples to generate and score on each call.
        Default: 10
    warm_up : int
        The number of randomly-generated samples to evaluate prior to running
        Bayesian optimization. Default: 20
    
    Other Parameters
    ----------------
    gp_kws
        Additional arguments to be passed to the Gaussian Process regressor.
        For details, see https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html#sklearn.gaussian_process.GaussianProcessRegressor
    """
    def __init__(self, n_samples=10, warm_up=20, **gp_kws):
        super().__init__(warm_up)
        self.n_samples = n_samples
        
        if 'n_samples' in gp_kws:
            del gp_kws['n_samples']
        if 'warm_up' in gp_kws:
            del gp_kws['warm_up']
        
        self.gp_kws = gp_kws

    def generate(self, trial_data, domains):
        """Generate a set of hyperparameters.

        Parameters
        ----------
        trial_data : array_like
            A 2-d numpy array where each row is one completed trial
            (hyperparameter set) and each column corresponds to one
            hyperparameter domain (always in the same order) with the
            objective value of the trial in the last column.
        domains : list of pyrameter.domain.base.Domain
            The domains from which hyperparameters were generated. These
            are provided in the same order as the columns in ``trial_data``.
        
        Returns
        -------
        array_like
            A 1-d list or array of new hyperparameter values with one element
            per hyperparameter domain in the same order as the columns in
            ``trial_data``.
        """
        features, losses = trial_data[-100:, :-1], trial_data[-100:, -1].reshape(-1, 1)
        params = []

        losses = StandardScaler().fit_transform(losses)

        # for j in range(len(space.domains)):
        # If no kernel is provided in the arguments, set the kernel to be a
        # default Matern
        if 'kernel' not in self.gp_kws:
            self.gp_kws['kernel'] = Matern()

        scaler = StandardScaler()
        x = scaler.fit_transform(features, losses)

        # Set up and train the Gaussian process regressor
        gp = GaussianProcessRegressor(n_restarts_optimizer=20, **self.gp_kws)
        gp.fit(x, losses)

        # Generate a number of candidate hyperparameter values.
        potential_params = np.zeros((self.n_samples, features.shape[1]))
        for i in range(self.n_samples):
            potential_params[i] += np.array([d.generate() for d in domains])
        scaled_params = scaler.transform(potential_params)

        # Compute the expected improvement of each candidate as a function of
        # the best-observed performance and the expectation and variance of the
        # predicted scores.
        mu, sigma = gp.predict(scaled_params, return_std=True)
        mu = mu.ravel()
        best = np.min(losses)
        with np.errstate(divide='ignore'):
            gamma = (best - mu) / sigma
        ei = (mu * (gamma * scipy.stats.norm.cdf(gamma))) + scipy.stats.norm.pdf(gamma)
        ei[sigma == 0] = 0  # sigma == 0 leads to NaNs in ei; handle it here

        params = potential_params[np.argmax(ei)]

        return params
