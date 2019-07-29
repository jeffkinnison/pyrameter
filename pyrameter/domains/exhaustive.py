"""Hyperparameter domain for exhaustive grid search.

Classes
-------
ExhaustiveDomain
    Discrete/categorical domain for exhaustive grid search.
"""

from collections import Sequence

from pyrameter.domains.discrete import DiscreteDomain


class ExhaustiveDomain(DiscreteDomain):
    """Discrete/categorical domain for exhaustive grid search.

    Parameters
    ----------
    name : str
        Name of the domain.
    domain : list
        The grid to search.

    Notes
    -----
    Instead of using internal tracking to determine which part of the grid to
    search, this domain is a placeholder used to spawn multiple search space
    graphs. As of now, it is not directly used to generate values.
    """

    def __init__(self, name, domain):
        super(ExhaustiveDomain, self).__init__(name, domain)

    @classmethod
    def from_json(cls, obj):
        domain = cls(obj['name'], obj['domain'])
        return domain

    @property
    def complexity(self):
        if self._complexity is None:
            try:
                self._complexity = 2 - (1 / len(self.domain))
            except ZeroDivisionError:
                self._complexity = 1
        return self._complexity

    def generate(self):
        raise NotImplementedError

    def to_json(self):
        jsonified = super(ExhaustiveDomain, self).to_json()
        return jsonified
