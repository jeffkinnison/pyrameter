"""Representation of a continuous hyperparameter domain.

Classes
-------
ContinuousDomain
    A continuous hyperparameter domain.
"""
import sys

import dill
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

    def __init__(self, *args, **kwargs):
        if len(args) == 0:
            raise ValueError('No domain provided.')
        elif len(args) == 1:
            super(ContinuousDomain, self).__init__()
            domain = args[0]
            args = args[1:]
        else:
            if isinstance(args[0], str) and isinstance(args[1], str):
                super(ContinuousDomain, self).__init__(args[0])
                domain = args[1]
                args = args[2:]
            else:
                super(ContinuousDomain, self).__init__()
                domain = args[0]
                args = args[1:]

        if isinstance(domain, str) and hasattr(scipy.stats, domain):
            self.domain = getattr(scipy.stats, domain)
        elif isinstance(domain, scipy.stats.rv_continuous):
            self.domain = domain
        else:
            raise ValueError('No domain provided.')

        callback = kwargs.pop('callback', None)
        seed = kwargs.pop('seed', None)

        self.callback = callback if callback is not None else lambda x: x
        self.seed = seed


        if seed is not None:
            if isinstance(seed, int):
                seed = np.random.RandomState(seed)
            kwargs['random_state'] = seed
        else:
            seed = np.random.RandomState(np.random.randint(int(2**32 - 1), dtype=np.uint32))
            kwargs['random_state'] = seed

        self.domain_args = args
        self.domain_kwargs = kwargs

    @property
    def complexity(self):
        if self._complexity is None:
            loc = self.domain_kwargs['loc'] if 'loc' in self.domain_kwargs \
                  else 0
            scl = self.domain_kwargs['scale'] if 'scale' in self.domain_kwargs \
                  else 1
            a, b = self.domain.interval(0.999, *self.domain_args,
                                        loc=loc, scale=scl)
            self._complexity = 2 + np.abs(b - a)
        return self._complexity

    @classmethod
    def from_json(cls, obj):
        if 'random_state' in obj['domain_kwargs']:
            rng = obj['domain_kwargs']['random_state']
            random_state = np.random.RandomState()
            random_state.set_state((rng[0], np.array(rng[1], dtype=np.uint32),
                                    rng[2], rng[3], rng[4]))
            del obj['domain_kwargs']['random_state']
        else:
            random_state = obj['domain_kwargs']['random_state']
            del obj['domain_kwargs']['random_state']
        domain = cls(obj['name'], obj['domain'], *obj['domain_args'],
                     dill.loads(obj['callback']), seed=random_state,
                     **obj['domain_kwargs'])
        return domain

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
