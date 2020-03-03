"""
"""

import numpy as np
from sklearn.mixture import GaussianMixture

from pyrameter.methods.random import random_search
from pyrameter.trial import Trial, TrialStatus


def tpe(space, best_split=0.2, n_samples=10, warm_up=20, **gmm_kws):
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
    n_complete = sum([1 for t in space.trials if t.status == TrialStatus.DONE])
    if n_complete < warm_up or n_complete % warm_up == 0:
        params = random_search(space)
    else:
        params = []

        # Collect all of the evaluated hyperparameter values and their
        # associated objective function value into a feature vector.
        data = space.to_array()
        features, losses = data[:, :-1], data[:, -1]

        # Sort the hyperparameters by their performance and split into
        # the "best" and "rest" performers.
        idx = np.argsort(losses)
        split = int(np.ceil(idx.shape[0] * best_split))
        losses = np.reshape(losses, (-1, 1))

        for j in range(len(space.domains)):

            # Model the objective function based on each feature.
            # for j in range(features.shape[0]):
            gmm_kws['n_components'] = 5
            l = GaussianMixture(**gmm_kws)
            gmm_kws['n_components'] = 5
            g = GaussianMixture(**gmm_kws)

            
            l.fit(features[idx[:split], j].reshape(-1, 1),
                    losses[idx[:split]])
            g.fit(features[idx[split:], j].reshape(-1, 1),
                    losses[idx[split:]])

            # Sample hyperparameter values from the "best" model and score
            # the samples with each model.
            samples, _ = l.sample(n_samples=n_samples)
            score_l = l.score_samples(samples)
            score_g = g.score_samples(samples)

            # Compute the expected improvement; i.e. maximize the l score
            # while minimizing the g score. Higher values are better.
            ei = score_l / score_g # best_split + (score_l / score_g * best_split)
            best = samples[np.argmax(np.squeeze(ei))]

            # Add the value with the best expected improvement
            domain = space.domains[j]
            params.append(domain.map_to_domain(best[0], bound=True))
            domain.current = params[-1]
    return params
