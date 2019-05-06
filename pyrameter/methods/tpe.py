"""Tree-structured Parzen Estimators for hyperparameter optimization.

Functions
---------
tpe
    Generate hyperparameters by optimizing for expected improvement.
"""

import numpy as np
from sklearn.mixture import GaussianMixture

from pyrameter.methods.random_search import random_search


def tpe(search_space, best_split=0.2, n_samples=10, warm_up=10, **gmm_kws):
    """Generate hyperparameters by optimizing for expected improvement.

    Parameters
    ----------
    search_space : pyrameter.db.SearchSpace
        The search space to draw values from.
    best_split : float, optional
        The percentage of hyperparameter values/performances that should be
        considered as "best" (the l mixture model). Must be in range (0, 1].
        Default 0.2.
    n_samples : int, optional
        The number of samples to generate from the l mixture model.
        Default 10.
    warm_up : int, optional
        The number of iterations of random search to run before starting TPE.
        Default 10.

    Other Parameters
    ----------------
    **gmm_kws
        Additional keyword arguments for Gaussian mixture model hyperparameters

    Returns
    -------
    hyperparameters : dict
        The set of hyperparameters generated from this search space.
    """
    n_results = len([r for r in search_space.results if r.loss is not None])
    if n_results < warm_up or n_results % warm_up == 0:
        hyperparameters = search_space.generate()
    else:
        vec = search_space.results_vector()

        features, losses = np.copy(vec[:, :-1]), np.copy(vec[:, -1])

        idx = np.argsort(losses, axis=0)
        split = int(np.ceil(idx.shape[0] * best_split))
        losses = np.reshape(losses, (-1, 1))

        l = GaussianMixture(**gmm_kws)  # "best" hyperparameters
        g = GaussianMixture(**gmm_kws)  # "rest" hyperparameters

        # Fit to the evaluated hyperparameters and objectives
        l.fit(features[idx[:split]], losses[idx[:split]])
        g.fit(features[idx[split:]], losses[idx[split:]])

        samples, _ = l.sample(n_samples=n_samples)
        score_l = l.score(samples)
        score_g = g.score(samples)

        expected_improvement = score_l / score_g
        hyperparameters = \
            samples[np.argmax(np.squeeze(expected_improvement).ravel())]


    return hyperparameters


# class TPEModel(RandomSearchModel):
#     """Search hyperparameters using Tree-Structured Parzen Estimators.
#
#     TPE uses a pair of Gaussian mixture models to models the opjective
#     function as a function of the best (l GMM) and rest (g GMM)
#     evaluted hyperparameters. Hyperparameters are implicitly separated
#     into disjoint trees prior to running TPE, and the two GMMs are
#     fit to a single tree at a time.
#
#     Parameters
#     ----------
#     id : str, optional
#     domains : list of `pyrameter.Domain`, optional
#     results : list of `pyrameter.models.Result`, optional
#     update_complexity : bool, optional
#     priority_update_freq : int, optional
#     best_split : float, optional
#         The percentage of hyperparameter values/performances that should be
#         considered as "best" (the l mixture model). Must be in range (0, 1].
#         Default 0.2.
#     n_samples : int, optional
#         The number of samples to generate from the l mixture model.
#         Default 10.
#     warm_up : int, optional
#         The number of iterations of random search to run before starting TPE.
#         Default 10,
#
#     Other Parameters
#     ----------------
#     **gmm_kws
#         Keyword args for parameterizing the Gaussian mixture models.
#
#     Notes
#     -----
#     Integer values (e.g., integer domains, indices into discrete domains) are
#     modeled by rounding off to the nearest integer. This is handled by
#     `pyrameter.DiscreteDomain`.
#
#     See Also
#     --------
#     `pyrameter.models.Model`
#     `pyrameter.models.RandomSearchModel`
#
#     """
#
#     TYPE = 'tpe'
#
#     def __init__(self, id=None, domains=None, results=None,
#                  update_complexity=True, priority_update_freq=10,
#                  best_split=0.2, n_samples=10, warm_up=10, **gmm_kws):
#         super(TPEModel, self).__init__(id=id,
#                                        domains=domains,
#                                        results=results,
#                                        update_complexity=update_complexity,
#                                        priority_update_freq= \
#                                             priority_update_freq)
#         self.gmm_kws = gmm_kws
#         self.best_split = best_split
#         self.n_samples = n_samples
#         self.warm_up = warm_up
#
#     def generate(self):
#         # Warm up with random search and inject new random search
#         # hyperparameters at an interval. This attempts to prevent TPE from
#         # converging too quickly.
#         if len(self.results) < self.warm_up or len(self.results) % self.warm_up == 0:
#             params = super(TPEModel, self).generate()
#         else:
#             params = np.zeros((len(self.domains),))
#
#             # Collect all of the evaluated hyperparameter values and their
#             # associated objective function value into a feature vector.
#             vec = self.results_to_feature_vector()
#             features, losses = np.copy(vec[:, :-1]), np.copy(vec[:, -1])
#             features = features.T
#
#             # Sort the hyperparameters by their performance and split into
#             # the "best" and "rest" performers.
#             idx = np.argsort(losses, axis=0)
#             split = int(np.ceil(idx.shape[0] * self.best_split))
#             losses = np.reshape(losses, (-1, 1))
#
#             # Model the objective function based on each feature.
#             for j in range(features.shape[0]):
#                 l = GaussianMixture(**self.gmm_kws)  # "best" hyperparameters
#                 g = GaussianMixture(**self.gmm_kws)  # "rest" hyperparameters
#                 l.fit(np.reshape(features[j, idx[:split]], (-1, 1)),
#                       losses[idx[:split]])
#                 g.fit(np.reshape(features[j, idx[split:]], (-1, 1)),
#                       losses[idx[split:]])
#
#                 # Sample hyperparameter values from the "best" model and score
#                 # the samples with each model.
#                 samples, _ = l.sample(n_samples=10)
#                 score_l = l.score(samples)
#                 score_g = g.score(samples)
#
#                 # Compute the expected improvement; i.e. maximize the l score
#                 # while minimizing the g score. Higher values are better.
#                 ei = score_l / score_g
#                 best = samples[np.argmax(np.squeeze(ei).ravel())]
#
#                 # Add the value with the best expected improvement
#                 domain = self.domains[j]
#                 params[j] += domain.map_to_domain(best[0], bound=True)
#
#         return params
