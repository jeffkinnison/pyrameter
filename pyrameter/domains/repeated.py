"""Representation of a continuous hyperparameter domain.

Classes
-------
RepeatedDomain
    A repeated hyperparameter domain structure.
"""
import copy
import sys

import dill
import numpy as np

from pyrameter.domains.base import Domain
from pyrameter.domains.constant import ConstantDomain
from pyrameter.domains.continuous import ContinuousDomain
from pyrameter.domains.discrete import DiscreteDomain
from pyrameter.domains.joint import JointDomain
from pyrameter.domains.sequence import SequenceDomain


class RepeatedDomain(Domain):
    """A repeated hyperparameter domain structure.

    In some cases, it is desirable to repeat a hyperparameter domain or scope,
    for example searching over neural network layers. This domain sets up 

    Parameters
    ----------
    name : str
        The name of this hyperparameter domain.
    domain : str or scipy.stats.rv_continuous
        The name of a continuous distribution defined in the scipy.stats module
        or a continuous distribution itself. Note: using frozen distributions
        will result in all domains using the same seed.
    repetitions: int
        The number of times to repeat ``domain``.
    split : bool, optional
        If True, split into ``repetitions`` domains of length
        [1, 2, 3, ..., ``repetitions``]. If False, acts the same as
        `pyramter.domains.SequentialDomain` during sampling. Default: True.
    callback : callable, optional
        An optional callback to run on generated hyperparameter values, e.g. to
        scale or otherwise modify the value.
    seed : int or numpy.random.RandomState, optional
        The random seed or random state to use to generate values.

    Attributes
    ----------
    """
    
    def __init__(self, *args, **kwargs):
        if len(args) < 2:
            raise ValueError('No domain or number of repetitions provided')
        if len(args) == 2:
            super(RepeatedDomain, self).__init__()
            domain = args[0]
            repetitions = args[1]
        else:
            super(RepeatedDomain, self).__init__(args[0])
            domain = args[1]
            repetitions = args[2]

        if isinstance(domain, dict):
            domain = JointDomain(**domain)
        elif isinstance(domain, list):
            domain = DiscreteDomain(domain)
        elif isinstance(domain, tuple):
            domain = SequenceDomain(domain)
        elif isinstance(domain, Domain):
            domain = domain
        elif hasattr(domain, '__class__') and 'Specification' in str(domain.__class__):
            domain = domain
        else:
            domain = ConstantDomain(domain)

        self.domain = [copy.deepcopy(domain) for _ in range(repetitions)]
        for i, d in enumerate(self.domain):
            d.name = f'{self.name}___{i}'
        self.repetitions = repetitions

        split = kwargs.pop('split', True)
        callback = kwargs.pop('callback', None)

        self.should_split = split
        self.callback = callback if callback is not None else lambda x: x

    @property
    def complexity(self):
        if self._complexity is None:
            self._complexity = np.product([d.complexity for d in self.domain])
        return self._complexity

    @classmethod
    def from_json(cls, obj):
        domain = cls(obj['name'], Domain.from_json(obj['domain']),
                     obj['repetitions'], split=obj['split'],
                     callback=dill.loads(obj['callback']))
        return domain

    def generate(self):
        """Generate a hyperparameter value from this domain."""
        return tuple([self.callback(d.generate()) for d in self.domain])

    def split(self):
        if self.should_split:
            return [RepeatedDomain(self.name, self.domain[0], i) for i in range(1, self.repetitions)]
        else:
            return self


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
            'domain': self.domain[0].to_json(),
            'repetitions': self.repetitions,
            'callback': dill.dumps(self.callback)
        })
