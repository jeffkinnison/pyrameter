"""Spearmint-style gaussian process-based Bayesian optimization.

Functions
---------
bayes_opt
    Spearmint-style gaussian process-based Bayesian optimization.
"""

import numpy as np
import scipy.stats
from sklearn.ensemble import RandomForestRegressor

from pyrameter.methods.method import Method


class SMAC(Method):
    """SMAC-style random forest bayesian optimization.

    Parameters
    ----------
    space : pyrameter.searchspace.SearchSpace
    """
    def __init__(self, n_samples=20, warm_up=10, **rf_kws):
        super().__init__(warm_up=warm_up)
        self.n_samples = n_samples
        self.rf_kws = rf_kws
    
    def generate(self, trial_data, domains):
        # Put the space's evaluated hyperparameters and result into arrays.
        features, losses = trial_data[:, :-1], trial_data[:, -1].ravel()
        params = []

        # Set up and train the Gaussian process regressor
        rf = RandomForestRegressor(**self.rf_kws)
        rf.fit(features, losses)

        # Generate a number of candidate hyperparameter values.
        potential_params = np.zeros((self.n_samples, features.shape[1]), dtype=np.float64)
        for i in range(self.n_samples):
            potential_params[i] += np.array([d.generate() for d in domains])

        # Compute the expected improvement of each candidate as a function of
        # the best-observed performance and the expectation and variance of the
        # predicted scores.
        preds = np.log(np.stack([t.predict(potential_params) for t in rf.estimators_], axis=0).T)

        mu = np.mean(preds, axis=1).ravel()
        sigma = np.var(preds, axis=1).ravel()
        best = np.min(losses)

        v = (np.log(best) - mu) / np.sqrt(sigma)
        left = (best * scipy.stats.norm.cdf(v))
        right = np.exp((0.5 * sigma) + mu) * scipy.stats.norm.cdf(v - np.sqrt(sigma))
        ei = left - right

        # Return the candidate with the best expected improvement
        params = potential_params[np.argmax(ei, axis=0)]

        return params
