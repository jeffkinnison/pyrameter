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

from pyrameter.methods.random import random_search
from pyrameter.trial import Trial, TrialStatus


def bayes(space, n_samples=500, warm_up=20, **gp_kws):
    """Spearmint-style gaussian process-based Bayesian optimization.

    Parameters
    ----------
    space : pyrameter.searchspace.SearchSpace
    """
    # Warm up with a number of random search results, and seed a with more
    # random searches at an interval throughout the search.
    completed = sum([1 for t in space.trials if t.status == TrialStatus.DONE])
    if completed == 0 or completed < warm_up:  # or completed % warm_up == 0:
        params = random_search(space)
    else:
        # Put the space's evaluated hyperparameters and result into arrays.
        data = space.to_array()
        features, losses = data[:, :-1], data[:, -1].reshape(-1, 1)
        params = []

        losses = StandardScaler().fit_transform(losses)

        for j in range(len(space.domains)):
            # If no kernel is provided in the arguments, set the kernel to be a
            # default Matern
            if 'kernel' not in gp_kws:
                gp_kws['kernel'] = Matern()

            if 'n_samples' in gp_kws:
                del gp_kws['n_samples']
            if 'warm_up' in gp_kws:
                del gp_kws['warm_up']

            scaler = StandardScaler()

            x = scaler.fit_transform(features[:, j].reshape(-1, 1), losses)

            # Set up and train the Gaussian process regressor
            gp = GaussianProcessRegressor(n_restarts_optimizer=20, **gp_kws)
            gp.fit(x, losses)

            # Generate a number of candidate hyperparameter values.
            potential_params = np.zeros((n_samples, 1))
            for i in range(n_samples):
                potential_params[i, 0] += space.domains[j].generate()
            potential_params = scaler.transform(potential_params)

            # Compute the expected improvement of each candidate as a function of
            # the best-observed performance and the expectation and variance of the
            # predicted scores.
            mu, sigma = gp.predict(potential_params, return_std=True)
            mu = mu.ravel()
            best = np.min(losses)
            with np.errstate(divide='ignore'):
                gamma = (best - mu) / sigma
            ei = (mu * (gamma * scipy.stats.norm.cdf(gamma))) + scipy.stats.norm.pdf(gamma)
            ei[sigma == 0] = 0  # sigma == 0 leads to NaNs in ei; handle it here

            # Return the candidate with the best expected improvement
            domain = space.domains[j]
            param_val = potential_params[np.argmax(ei)]
            param_val = scaler.inverse_transform([[param_val]])[0, 0]
            params.append(domain.map_to_domain(float(param_val), bound=True))
            domain.current = params[-1]

    return params
