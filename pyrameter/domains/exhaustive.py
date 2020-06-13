"""Hyperparameter domain for exhaustive grid search.

Classes
-------
ExhaustiveDomain
    Discrete/categorical domain for exhaustive grid search.
"""

from collections import Sequence

from pyrameter.domains.base import Domain


class ExhaustiveDomain(Domain):
    """Discrete/categorical domain for exhaustive grid search.

    Parameters
    ----------
    name : str
        Name of the domain.
    domain : list
        The grid to search.
    """

    def __init__(self, *args, **kwargs):
        if len(args) >= 2:
            super(ExhaustiveDomain, self).__init__(args[0])
            self.domain = args[1]
        elif len(args) == 1:
            super(ExhaustiveDomain, self).__init__()
            self.domain = args[0]
        else:
            raise ValueError('No domain provided.')

        if isinstance(self.domain, range):
            self.domain = list(self.domain)

        if not isinstance(self.domain, list):
            self.domain = [self.domain]

        self._index = 0

    @classmethod
    def from_json(cls, obj):
        domain = cls(obj['name'], obj['domain'])
        domain._index = obj.get('index', 0)
        return domain

    @property
    def complexity(self):
        if self._complexity is None:
            try:
                self._complexity = 2 - (1 / len(self.domain))
            except (TypeError, ZeroDivisionError):
                self._complexity = 1
        return self._complexity

    def generate(self):
        if len(self.domain) > 0:
            val = self.domain[self._index]
            self._index += 1
            
            if self._index >= len(self.domain):
                self._index = 0
        else:
            val = None
        
        return val

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
        jsonified = super(ExhaustiveDomain, self).to_json()
        jsonified['domain'] = list(self.domain)
        jsonified['index'] = self._index
        return jsonified
