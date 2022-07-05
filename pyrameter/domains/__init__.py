"""Hyperparameter search domain definitions.

Classes
-------
Domain
    Base class for hyperparameter domains.
ConditionalDomain
    A hyperparameter domain which is dependent upon another.
ConstantDomain
    A singleton hyperparameter domain.
ContinuousDomain
    A continuous hyperparameter domain.
DependentDomain
    A hyperparameter domain which is dependent upon another.
DiscreteDomain
    A Discrete hyperparameter domain.
SequenceDomain
    Multiple ordered hyperparameter domains.
"""

from .base import Domain
from .conditional import ConditionalDomain
from .constant import ConstantDomain
from .continuous import ContinuousDomain
from .linked import DependentDomain
from .discrete import DiscreteDomain
from .exhaustive import ExhaustiveDomain
from .joint import JointDomain
# from .searchspace import SearchSpace
from .repeated import RepeatedDomain
from .sequence import SequenceDomain

__all__ = ["Domain", "ConditionalDomain", "ConstantDomain", "ContinuousDomain",
           "DependentDomain", "DiscreteDomain", "ExhaustiveDomain",
           "JointDomain", "RepeatedDomain", "SequenceDomain"]
