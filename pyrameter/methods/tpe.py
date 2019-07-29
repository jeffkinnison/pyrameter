"""
"""

import numpy as np
from sklearn.mixture import GaussianMixture

from pyrameter.methods.random import random_search
from pyrameter.trial import Trial


def tpe(space, best_split=0.2, n_samples=10, warm_up=10, **gmm_kws):
    """Tree-structured Parzen Enstimators for generating hyperparameters.

    Parameters
    ----------
    space : pyrameter.domains.SearchSpace
        The space to generate values from.
    best_split : float in [0, 1]
        The percentage of results to use for the top-k mixture model.
    n_samples : int
        The number of candidate samples to generate.
    warm_up : int
        The number of random search iterations to use to seed TPE.

    Other Parameters
    ----------------
    **gmm_kws
        Additional keyword arguments to parameterize the Gaussian Mixture
        Models.

    Returns
    -------
    values : array-like
        The array of hyperparameter values with the highest expected
        improvement from among the candidate ``n_samples``.
    """
    # Warm up with random search and inject new random search
    # hyperparameters at an interval. This attempts to prevent TPE from
    # converging too quickly.
    if len(space.objective) < warm_up or len(space.objective) % warm_up == 0:
        params = random_search(space)
    else:
        params = []

        # Collect all of the evaluated hyperparameter values and their
        # associated objective function value into a feature vector.
        features = space.to_array().T
        losses = np.array(space.objective)

        # Sort the hyperparameters by their performance and split into
        # the "best" and "rest" performers.
        idx = np.argsort(losses, axis=0)
        split = int(np.ceil(idx.shape[0] * best_split))
        losses = np.reshape(losses, (-1, 1))

        # Model the objective function based on each feature.
        for j in range(features.shape[0]):
            l = GaussianMixture(**gmm_kws)
            g = GaussianMixture(**gmm_kws)
            l.fit(np.reshape(features[j, idx[:split]], (-1, 1)),
                  losses[idx[:split]])
            l.fit(np.reshape(features[j, idx[split:]], (-1, 1)),
                  losses[idx[split:]])

            # Sample hyperparameter values from the "best" model and score
            # the samples with each model.
            samples, _ = l.sample(n_samples=n_samples)
            score_l = l.score(samples)
            score_g = g.score(samples)

            # Compute the expected improvement; i.e. maximize the l score
            # while minimizing the g score. Higher values are better.
            ei = score_l / score_g
            best = samples[np.argmax(np.squeeze(ei).ravel())]

            # Add the value with the best expected improvement
            domain = space.nodes[j]
            params.append(domain.map_to_domain(best[0]), bound=True)

            params = Trial(space, hyperparameters=params)
            space.results.append(params)
    return params
