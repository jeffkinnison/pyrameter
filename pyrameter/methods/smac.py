"""Spearmint-style gaussian process-based Bayesian optimization.

Functions
---------
bayes_opt
    Spearmint-style gaussian process-based Bayesian optimization.
"""

import numpy as np
import scipy.stats
from sklearn.ensemble import RandomForestRegressor

from pyrameter.methods.random import random_search
from pyrameter.trial import Trial, TrialStatus


def smac(space, n_samples=100, warm_up=10, **gp_kws):
    """SMAC-style random forest bayesian optimization.

    Parameters
    ----------
    space : pyrameter.searchspace.SearchSpace
    """
    # Warm up with a number of random search results, and seed a with more
    # random searches at an interval throughout the search.
    completed = sum([1 for t in space.trials if t.status == TrialStatus.DONE])
    if completed < warm_up or completed % warm_up == 0:
        params = random_search(space)
    else:
        # Put the space's evaluated hyperparameters and result into arrays.
        data = space.to_array()
        features, losses = data[:, :-1], data[:, -1].reshape(-1, 1)
        params = []

        for j in range(len(space.domains)):
            # Set up and train the Gaussian process regressor
            rf = RandomForestRegressor(**gp_kws)
            rf.fit(features[:, j].reshape(-1, 1), losses)

            # Generate a number of candidate hyperparameter values.
            potential_params = np.zeros((n_samples, 1))
            for i in range(n_samples):
                potential_params[i] += space.domains[j].generate()

            # Compute the expected improvement of each candidate as a function of
            # the best-observed performance and the expectation and variance of the
            # predicted scores.
            preds = [t.predict(potential_params) for t in rf.estimators_]
            mu = np.mean(preds)
            sigma = np.mean(preds)
            best = np.min(losses)
            with np.errstate(divide='ignore'):
                gamma = (mu - best) / sigma
            ei = (mu - gamma) * scipy.stats.norm.cdf(gamma) + \
                sigma * scipy.stats.norm.pdf(gamma)
            ei[sigma == 0] = 0  # sigma == 0 leads to NaNs in ei; handle it here

            # Return the candidate with the best expected improvement
            params.append(potential_params[np.argmax(ei, axis=1)[0]])

            domain = space.domains[j]
            params.append(domain.map_to_domain(float(params[j]), bound=True))
            domain.current = params[-1]

    return params
