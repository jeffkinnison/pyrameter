"""Representation of a sequence of hyperparameter domains.

Classes
-------
SequenceDomain
    Multiple hyperparameter domains grouped together.
"""

import dill
import numpy as np

from pyrameter.domains.base import Domain
from pyrameter.domains.constant import ConstantDomain
from pyrameter.domains.continuous import ContinuousDomain
from pyrameter.domains.discrete import DiscreteDomain
from pyrameter.domains.joint import JointDomain


class SequenceDomain(Domain):
    """Multiple ordered hyperparameter domains.

    Parameters
    ----------
    name : str
        The name of this hyperparameter domain.
    domain : tuple of subclass of pyrameter.domains.base.Domain
        The name of a continuous distribution defined in the scipy.stats module
        or a continuous distribution itself. Note: using frozen distributions
        will result in all domains using the same seed.
    callback : callable, optional
        An optional callback to run on generated hyperparameter values, e.g. to
        scale or otherwise modify the value.

    """

    def __init__(self, *args, **kwargs):
        if len(args) >= 2:
            super(SequenceDomain, self).__init__(args[0])
            domain = args[1]
        elif len(args) == 1:
            super(SequenceDomain, self).__init__()
            domain = args[0]
        else:
            raise ValueError('No domain provided.')

        if not isinstance(domain, tuple):
            domain = tuple([domain])

        adjusted_domains = []
        for d in domain:
            if isinstance(d, dict):
                adjusted_domains.append(JointDomain(**d))
            elif isinstance(d, list):
                adjusted_domains.append(DiscreteDomain(d))
            elif isinstance(d, tuple):
                adjusted_domains.append(SequenceDomain(d))
            elif isinstance(d, Domain):
                adjusted_domains.append(d)
            else:
                adjusted_domains.append(ConstantDomain(d))

        self.domain = tuple(adjusted_domains)

        callback = kwargs.pop('callback', None)
        self.callback = callback if callback is not None else lambda x: x

    @property
    def complexity(self):
        if self._complexity is None:
            self._complexity = np.product([d.complexity for d in self.domain])
        return self._complexity

    @classmethod
    def from_json(cls, obj):
        domain = cls(obj['name'], [Domain.from_json(d) for d in obj['domains']],
                 callback=dill.loads(obj['callback']))

    def generate(self):
        """Generate a hyperparameter value from this domain."""
        return tuple([self.callback(d.generate()) for d in self.domain])

    def to_index(self, value):
        """Convert a value to its index in the domain."""
        try:
            idx = tuple([d.to_index(value) for d in self.domain])
        except ValueError:
            idx = None
        return idx

    def to_json(self):
        jsonified = super().to_json()
        jsonified.update({
            'domain': tuple([d.to_json() for d in self.domain]),
            'callback': dill.dumps(self.callback)
        })
