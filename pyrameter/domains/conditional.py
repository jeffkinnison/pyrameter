"""Representation of a discrete hyperparameter domain conditioned upon another.

Classes
-------
DependentDomain
    A hyperparameter domain which is dependent upon another.
"""

from pyrameter.domains.base import Domain


class ConditionalDomain(Domain):
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
            super(ConditionalDomain, self).__init__(args[0])
            self.domain = args[1]
        elif len(args) == 1:
            super(ConditionalDomain, self).__init__()
            self.domain = args[0]
        else:
            raise ValueError('No domain provided.')

        callback = kwargs.pop('callback', None)

        self.callback = callback if callback is not None else lambda x: x

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
