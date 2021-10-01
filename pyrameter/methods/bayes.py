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
    def __init__(self, n_samples=10, warm_up=20, **gp_kws):
        super().__init__(warm_up)
        self.n_samples = n_samples
        
        if 'n_samples' in gp_kws:
            del gp_kws['n_samples']
        if 'warm_up' in gp_kws:
            del gp_kws['warm_up']
        
        self.gp_kws = gp_kws

    def generate(self, trial_data, domains):
        """Spearmint-style gaussian process-based Bayesian optimization.

        Parameters
        ----------
        space : pyrameter.searchspace.SearchSpace
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

        # # Return the candidate with the best expected improvement
        # domain = space.domains[j]
        # param_val = potential_params[np.argmax(ei)]
        # param_val = scaler.inverse_transform([[param_val]])[0, 0]
        # params.append(domain.map_to_domain(float(param_val), bound=True))
        # domain.current = params[-1]

        params = potential_params[np.argmax(ei)]

        return params
