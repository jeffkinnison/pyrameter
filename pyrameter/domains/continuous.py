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

        if seed is not None:
            if isinstance(seed, int):
                seed = np.random.RandomState(seed)
            domain_kwargs['random_state'] = seed

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

    def map_to_domain(self, value, bound=False):
        if bound:
            try:
                pdf_kwargs = {k: v for k, v in self.domain_kwargs.items()
                              if k != 'random_state'}
                prob = self.domain.pdf(
                    value, *self.domain_args, **pdf_kwargs)
                if prob == 0:
                    value = None
            except ValueError:
                value = None
        return value

    def to_index(self, value, bound=False):
        if bound:
            try:
                pdf_kwargs = {k: v for k, v in self.domain_kwargs.items()
                              if k != 'random_state'}
                prob = self.domain.pdf(
                    value, *self.domain_args, **pdf_kwargs)
                if prob == 0:
                    value = None
            except ValueError:
                value = None
        return value

    def to_json(self):
        jsonified = super(ContinuousDomain, self).to_json()
        jsonified.update({
            'domain': self.domain.name,
            'domain_args': self.domain_args,
            'domain_kwargs': {}
        })

        for key, val in self.domain_kwargs.items():
            if key == 'random_state' and isinstance(val, np.random.RandomState):
                val = list(val.get_state())
                val[1] = list(val[1])
            jsonified['domain_kwargs'][key] = val

        return jsonified
