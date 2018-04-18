from pyrameter.models.random_search import RandomSearchModel

import numpy as np
from sklearn.mixture import GaussianMixture


class TPEModel(RandomSearchModel):
    def __init__(self, id=None, domains=None, results=None,
                 update_complexity=True, priority_update_freq=10,
                 best_split=0.2, n_samples=10, warm_up=10, **gmm_kws):
        super(TPEModel, self).__init__(id=id,
                                       domains=domains,
                                       results=results,
                                       update_complexity=update_complexity,
                                       priority_update_freq= \
                                            priority_update_freq)
        self.gmm_kws = gmm_kws
        self.best_split = best_split
        self.n_samples = n_samples
        self.warm_up = warm_up

    def generate(self):
        if len(self.results) < self.warm_up or len(self.results) % self.warm_up == 0:
            params = super(TPEModel, self).generate()
        else:
            params = {}

            vec = self.results_to_feature_vector()
            features, losses = np.copy(vec[:, :-1]), np.copy(vec[:, -1])
            features = features.T
            idx = np.argsort(losses, axis=0)
            split = int(np.ceil(idx.shape[0] * self.best_split))
            losses = np.reshape(losses, (-1, 1))

            for j in range(features.shape[0]):
                l = GaussianMixture(**self.gmm_kws)
                g = GaussianMixture(**self.gmm_kws)
                l.fit(np.reshape(features[j, idx[:split]], (-1, 1)),
                      losses[idx[:split]])
                g.fit(np.reshape(features[j, idx[split:]], (-1, 1)),
                      losses[idx[split:]])

                samples, _ = l.sample(n_samples=10)
                score_l = l.score(samples)
                score_g = g.score(samples)

                ei = score_l / score_g
                best = samples[np.argmax(np.squeeze(ei).ravel())]

                domain = self.domains[j]
                path = domain.path.split('/')
                curr = params
                for p in path[:-1]:
                    if p not in curr:
                        curr[p] = {}
                    curr = curr[p]
                curr[path[-1]] = domain.map_to_domain(best[0], bound=True)

        return params
