"""Representation of a continuous hyperparameter domain.

Classes
-------
ContinuousDomain
    A continuous hyperparameter domain.
"""

import numpy as np
import scipy.stats

from pyrameter.domains.base import Domain


class ContinuousDomain(Domain):
    """A continuous hyperparameter domain.

    Parameters
    ----------
    name : str
        The name of this hyperparameter domain.
    domain : str or scipy.stats.rv_continuous
        The name of a continuous distribution defined in the scipy.stats module
        or a continuous distribution itself. Note: using frozen distributions
        will result in all domains using the same seed.
    callback : callable, optional
        An optional callback to run on generated hyperparameter values, e.g. to
        scale or otherwise modify the value.
    seed : int or numpy.random.RandomState, optional
        The random seed or random state to use to generate values.

    """

    def __init__(self, name, domain, *domain_args, callback=None, seed=None,
                 **domain_kwargs):
        super(ContinuousDomain, self).__init__(name)
        try:
            self.domain = getattr(scipy.stats, domain)
        except AttributeError:
            self.domain = domain

        self.callback = callback if callback is not None else lambda x: x
        self.seed = seed

        domain_kwargs.pop('callback', None)
        domain_kwargs.pop('seed', None)

        self.domain_args = domain_args
        self.domain_kwargs = domain_kwargs

    @property
    def complexity(self):
        if self._complexity is None:
            a, b = self.domain.interval(0.999, *self.domain_args,
                                        **self.domain_kwargs)
            self._complexity = 2 + np.abs(b - a)
        return self._complexity

    def generate(self):
        """Generate a hyperparameter value from this domain."""
        return self.callback(
            self.domain.rvs(*self.domain_args, **self.domain_kwargs))
