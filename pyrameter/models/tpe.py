from pyrameter.models.model import Model

import numpy as np
from sklearn.mixture import GaussianMixture


class TPEModel(Model):
    def __init__(self, id=None, domains=None, results=None,
                 update_complexity=True, priority_update_freq=10,
                 best_split=0.2, n_samples=10, **gmm_kws):
        super(TPEModel, self).__init__(id=id,
                                       domains=domains,
                                       results=results,
                                       update_complexity=update_complexity,
                                       priority_update_freq= \
                                            priority_update_freq)
        self.l = GaussianMixture(**gmm_kws)
        self.g = GaussianMixture(**gmm_kws)
        self.best_split = best_split
        self.n_samples = n_samples

    def generate(self):
        vec = self.__results_to_feature_vector()
        features, losses = np.copy(vec[:, :-1]), np.copy(vec[:, -1])
        idx = np.argsort(losses, axis=0)
        split = int(np.ceil(idx.shape[0] * self.best_split))
        losses = np.reshape(losses, (-1, 1))
        self.l.fit(features[idx[:split]], losses[idx[:split]])
        self.g.fit(features[idx[split:]], losses[idx[split:]])

        samples = self.l.sample(n_samples=10)
        score_l = self.l.score(samples)
        score_g = self.g.score(samples)

        ei = score_l / score_g
        best = samples[np.argmax(ei)]

        params = {}
        for i in range(len(self.domains)):
            domain = self.domains[i]
            path = domain.path.split('/')
            curr = params
            for p in path[:-1]:
                if p not in curr:
                    curr[p] = {}
                curr = curr[p]
            curr[path[-1]] = domain.map_to_domain(samples[i])

        return params
