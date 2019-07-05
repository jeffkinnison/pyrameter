"""Representation of a discrete hyperparameter domain dependent upon another.

Classes
-------
DependentDomain
    A hyperparameter domain which is dependent upon another.
"""

from pyrameter.domains.base import Domain


class DependentDomain(Domain):
    """A hyperparameter domain which is dependent upon another.

    Parameters
    ----------
    name : str
        The name of this hyperparameter domain.
    domain : subclass of pyrameter.domains.base.Domain
        The name of a continuous distribution defined in the scipy.stats module
        or a continuous distribution itself. Note: using frozen distributions
        will result in all domains using the same seed.
    callback : callable, optional
        An optional callback to run on generated hyperparameter values, e.g. to
        scale or otherwise modify the value.

    """

    def __init__(self, name, domain, callback=None):
        super(DependentDomain, self).__init__(name)

        if not isinstance(domain, Domain):
            raise InvalidDomainError()

        self.domain = domain

        self.callback = callback if callback is not None else lambda x: x

    def __ge__(self, other):
        if other is self.domain or other == self.domain:
            return True
        return hash(self) >= other(hash)

    def __gt__(self, other):
        if other is self.domain or other == self.domain:
            return True
        return hash(self) > other(hash)

    def __le__(self, other):
        if other is self.domain or other == self.domain:
            return True
        return hash(self) <= other(hash)

    def __lt__(self, other):
        if other is self.domain or other == self.domain:
            return True
        return hash(self) < other(hash)

    @property
    def complexity(self):
        if self._complexity is None:
            self._complexity = self.domain.complexity
        return self._complexity

    def generate(self):
        """Generate a hyperparameter value from this domain."""
        return self.callback(self.domain._current)

    def to_index(self, value):
        """Convert a value to its index in the domain."""
        try:
            idx = self.domain.to_index(value)
        except ValueError:
            idx = None
        return idx
