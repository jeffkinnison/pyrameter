"""Tree-structured Parzen Enstimators for generating hyperparameters.

Classes
-------
TPE

"""

import numpy as np
from sklearn.mixture import GaussianMixture

from pyrameter.methods.method import Method
from pyrameter.methods.random import RandomSearch
from pyrameter.trial import Trial, TrialStatus


class TPE(Method):
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
    def __init__(self, best_split=0.2, n_samples=10, warm_up=50, **gmm_kws):
        super().__init__(warm_up)

        self.best_split = best_split
        self.n_samples = n_samples
        self.gmm_kws = gmm_kws
    
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
        # Collect all of the evaluated hyperparameter values and their
        # associated objective function value into a feature vector.
        features, losses = trial_data[:, :-1], trial_data[:, -1]

        # Sort the hyperparameters by their performance and split into
        # the "best" and "rest" performers.
        idx = np.argsort(losses)
        split = int(np.ceil(idx.shape[0] * self.best_split))
        losses = np.reshape(losses, (-1, 1))

        params = []

        # for j in range(features.shape[1]):
        # Model the objective function based on each feature.
        self.gmm_kws['n_components'] = 5
        l = GaussianMixture(**self.gmm_kws)
        self.gmm_kws['n_components'] = 5
        g = GaussianMixture(**self.gmm_kws)

        l.fit(features[idx[:split]],
                losses[idx[:split]])
        g.fit(features[idx[split:]],
                losses[idx[split:]])

        # Sample hyperparameter values from the "best" model and score
        # the samples with each model.
        samples, _ = l.sample(n_samples=self.n_samples)
        score_l = l.score_samples(samples)
        score_g = g.score_samples(samples)

        # Compute the expected improvement; i.e. maximize the l score
        # while minimizing the g score. Higher values are better.
        ei = score_l / score_g # best_split + (score_l / score_g * best_split)
        best = samples[np.argmax(np.squeeze(ei))]

        # Add the value with the best expected improvement
        params = best
        return params
