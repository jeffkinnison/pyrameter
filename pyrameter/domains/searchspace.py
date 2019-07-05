"""Hierarchical domain organization.

Classes
-------
SearchSpace
    Hierarchical hyperparameter domain organization.
"""

import collections
import functools
import operator
import warnings

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

from pyrameter.domains.base import Domain


class SearchSpace(object):
    """Hierarchical hyperparameter domain organization.
    """

    def __init__(self, domains):
        self.objective = []
        self.results = []

        self.__complexity = None
        self.__uncertainty = None

        self.domains = domains if domains is not None else []
        domains.sort()

    @property
    def complexity(self):
        if self.__complexity is None:
            self.__complexity = functools.reduce(operator.mul, self.nodes, 1.0)
        return self.__complexity

    def register_result(self, result, objective_key='loss'):
        self.objective.append(result[objective_key])
        self.results.append(result)

    def to_array(self):
        out = np.zeros((len(self.domains), len(self.domains[0].history)),
              dtype=np.float32)

        for i, domain in enumerate(self.domains):
            out[i, :] += np.array([domain.to_index(d) for d in domain.history])
        return out

    def generate(self, method):
        return [d.generate() for d in self.domains]

    @property
    def uncertainty(self):
        if len(self.objective) > 10:
            uncertainty_array = self.to_array()
            labels = np.array(self.objective)

            split = int(np.floor(labels.shape * 0.8))

            scales = np.zeros(50)
            for i in range(50):
                indices = np.random.permutation(np.arange(labels.shape[0]))
                est = np.random.uniform(0.1, 2.0)
                gp = GaussianProcessRegressor(kernel=RBF(length_scale=est),
                                              alpha=1e-5)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    gp.fit(uncertainty_array[indices[:split]],
                           labels[indices[:split]])
                scales[i] += np.power(gp.kernel.theta[0], -1.0)

            self.__uncertainty = np.linalg.norm(scales.max() - scales.min())
        else:
            self.__uncertainty = 1

        return self.__uncertainty
