from arbor.models.model import Model

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor


class GPBayesModel(Model):
    def __init__(self, n_samples=10, **gp_kws):
        super(GPBayesModel, self).__init__(self)
        self.gp = GaussianProcessRegressor(**gp_kws)
        self.n_samples = n_samples

    def generate(self):
        vec = self.__results_to_feature_vector()
        features = np.copy(vec[:, :-1])
        losses = np.copy(vec[:, -1])
        losses = np.reshape(vec, (-1, 1))

        self.gp.fit(features, losses)

        potentials = np.zeros((n_samples, len(self.domains)))
        for i in range(n_samples):
            for j in range(len(self.domains)):
                potentials[i, j] = self.domains[j].generate(index=True)

        mu, sigma = gp.predict(potentials)
        best = np.max(losses)
        gamma = (best - mu) / sigma
        ei = sigma * (gamma * norm.cdf(gamma) + norm.pdf(gamma))

        best = potentials[np.argmax(ei)]

        params = {}
        for i in range(len(self.domains)):
            domain = self.domains[i]
            path = domain.path.split('/')
            curr = params
            for p in path[:-1]:
                if p not in curr:
                    curr[p] = {}
                curr = curr[p]
            curr[path[-1]] = domain.map_to_domain(params[i])

        return params
