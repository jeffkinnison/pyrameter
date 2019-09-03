"""Representation of a discrete hyperparameter domain.

Classes
-------
DiscreteDomain
    A discrete hyperparameter domain.
"""

from collections import Sequence

import dill
import numpy as np
import scipy.stats

from pyrameter.domains.base import Domain


class DiscreteDomain(Domain):
    """A Discrete hyperparameter domain.

    Parameters
    ----------
    name : str
        The name of this hyperparameter domain.
    domain : list
        A collection of values in the domain.
    callback : callable, optional
        An optional callback to run on generated hyperparameter values, e.g. to
        scale or otherwise modify the value.
    seed : int or numpy.random.RandomState, optional
        The random seed or random state to use to generate values.

    """

    def __init__(self, *args, **kwargs):
        if len(args) >= 2:
            super(DiscreteDomain, self).__init__(args[0])
            domain = args[1]
        elif len(args) == 1:
            super(DiscreteDomain, self).__init__()
            domain = args[0]
        else:
            raise ValueError('No domain provided.')

        callback = kwargs.pop('callback', None)
        seed = kwargs.pop('seed', None)

        if not isinstance(domain, Sequence) or isinstance(domain, (str, tuple)):
            domain = [domain]
        elif isinstance(domain, range):
            domain = list(domain)

        self.domain = list(domain)

        if isinstance(seed, int):
            seed = np.random.RandomState(seed)

        self.callback = callback if callback is not None else lambda x: x
        self.random_state = seed

    @property
    def complexity(self):
        if self._complexity is None:
            try:
                self._complexity = 2 - (1 / len(self.domain))
            except ZeroDivisionError:
                self._complexity = 1
        return self._complexity

    @classmethod
    def from_json(cls, obj):
        if 'random_state' in obj:
            rng = obj['domain_kwargs']['random_state']
            random_state = np.random.RandomState()
            random_state.set_state((rng[0], np.array(rng[1], dtype=np.uint32),
                                    rng[2], rng[3], rng[4]))
            del obj['random_state']
        else:
            random_state = obj['random_state']
            del obj['random_state']
        domain = cls(obj['name'], obj['domain'],
                     callback=dill.loads(obj['callback']), seed=random_state)
        return domain

    def generate(self):
        """Generate a hyperparameter value from this domain."""
        if len(self.domain) > 0:
            index = self.callback(
                scipy.stats.randint.rvs(0, len(self.domain),
                                        random_state=self.random_state))
            return self.domain[index]
        else:
            return None

    def map_to_domain(self, idx, bound=True):
        if bound:
            idx = int(round(idx))
            idx = min(len(self.domain) - 1, max(0, idx))
        elif not bound and idx < 0:
            idx = len(self.domain)
        try:
            val = self.domain[idx]
        except IndexError:
            val = None
        return val

    def to_index(self, value):
        """Convert a value to its index in the domain."""
        try:
            idx = self.domain.index(value)
        except ValueError:
            idx = None
        return idx

    def to_json(self):
        jsonified = super(DiscreteDomain, self).to_json()
        jsonified.update({
            'domain': list(self.domain)
        })
        if isinstance(self.random_state, np.random.RandomState):
            rs = self.random_state.get_state()
            jsonified.update({
                'random_state': [rs[0], list(rs[1]), rs[2], rs[3], rs[4]]
            })
        else:
            jsonified.update({'random_state': self.random_state})
        return jsonified
