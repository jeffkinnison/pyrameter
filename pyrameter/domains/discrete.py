"""Representation of a discrete hyperparameter domain.

Classes
-------
DiscreteDomain
    A discrete hyperparameter domain.
"""

from collections import Sequence

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

    def __init__(self, name, domain, callback=None, seed=None):
        super(DiscreteDomain, self).__init__(name)

        if not isinstance(domain, Sequence) or isinstance(domain, str):
            domain = [domain]

        self.domain = list(domain)

        self.callback = callback if callback is not None else lambda x: x
        self.seed = seed

    @property
    def complexity(self):
        if self._complexity is None:
            self._complexity = 2 - (1 / len(self.domain))
        return self._complexity

    def generate(self):
        """Generate a hyperparameter value from this domain."""
        index = self.callback(
            scipy.stats.randint.rvs(0, len(self.domain),
                                    random_state=self.seed))
        return self.domain[index]

    def map_to_value(self, idx, bound=True):
        if bound:
            idx = int(round(idx))
            idx = min(len(self.domain) - 1, max(0, idx))
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
