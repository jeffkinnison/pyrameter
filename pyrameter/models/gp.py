from pyrameter.models.random_search import RandomSearchModel

import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF


class GPBayesModel(RandomSearchModel):
    """Gaussian process-based hyperparameter optimizer.

    Based on the Spearmint

    Paramaters
    ----------
    id :
    """

    TYPE = 'gp'

    def __init__(self, id=None, domains=None, results=None,
                 update_complexity=True, priority_update_freq=10, n_samples=10,
                 warm_up=10, **gp_kws):
        super(GPBayesModel, self).__init__(id=id,
                                           domains=domains,
                                           results=results,
                                           update_complexity=update_complexity,
                                           priority_update_freq= \
                                                priority_update_freq)
        self.n_samples = n_samples
        self.warm_up = warm_up
        self.gp_kws = gp_kws
        if 'kernel' not in self.gp_kws:
            self.gp_kws['kernel'] = RBF()

    def generate(self):
        if len(self.results) < self.warm_up or len(self.results) % self.warm_up == 0:
            params = super(GPBayesModel, self).generate()
        else:
            vec = self.results_to_feature_vector()
            features, losses = np.copy(vec[:, :-1]), np.copy(vec[:, -1])
            #features = features.T
            losses = np.reshape(losses, (-1, 1))

            gp = GaussianProcessRegressor(**self.gp_kws)
            gp.fit(features, losses)

            potentials = np.zeros((self.n_samples, len(self.domains)))
            for i in range(self.n_samples):
                for j in range(len(self.domains)):
                    val = self.domains[j].generate(index=True)
                    if isinstance(val, tuple):
                        val = val[1]
                    potentials[i, j] += val

            mu, sigma = gp.predict(potentials, return_std=True)
            best = np.min(losses)
            with np.errstate(divide='ignore'):
                gamma = (mu - best) / sigma
            ei = (mu - gamma) * norm.cdf(gamma) + sigma * norm.pdf(gamma)
            ei[sigma == 0] = 0

            best = potentials[np.argmax(ei, axis=1)]

            params = np.zeros((len(self.domains),))
            for i in range(len(self.domains)):
                domain = self.domains[i]
                params[i] += domain.map_to_domain(best[i][0],
                                                  bound=True)

        return params
