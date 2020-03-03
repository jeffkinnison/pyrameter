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

    def __init__(self, *args, **kwargs):
        if len(args) >= 2:
            super(DependentDomain, self).__init__(args[0])
            domain = args[1]
        elif len(args) == 1:
            super(DependentDomain, self).__init__()
            domain = args[0]
        else:
            raise ValueError('No domain provided.')

        if not isinstance(domain, Domain):
            raise ValueError('{} is not a valid pyrameter Domain.'.format(domain))

        self.domain = domain

        callback = kwargs.pop('callback', None)
        self.callback = callback if callback is not None else lambda x: x

    def __ge__(self, other):
        if other is self.domain or other == self.domain:
            return True
        return super().__ge__(other)

    def __gt__(self, other):
        if other is self.domain or other == self.domain:
            return True
        return super().__gt__(other)

    def __le__(self, other):
        if other is self.domain or other == self.domain:
            return False
        return super().__le__(other)

    def __lt__(self, other):
        if other is self.domain or other == self.domain:
            return False
        return super().__lt__(other)

    @property
    def complexity(self):
        if self._complexity is None:
            self._complexity = self.domain.complexity
        return self._complexity

    def generate(self):
        """Generate a hyperparameter value from this domain."""
        return self.callback(self.domain.current)

    def to_index(self, value):
        """Convert a value to its index in the domain."""
        try:
            idx = self.domain.to_index(value)
        except ValueError:
            idx = None
        return idx
