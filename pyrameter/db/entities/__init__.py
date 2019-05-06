"""Data models for database interaction.

Classes
-------
Domain
    Data model of a hyperparameter search domain.
Result
    Data model of a hyperparameter evaluation result.
SearchSpace
    Data model of a joint search space (collection of Domains and Results).

"""

from .searchspace import SearchSpace
from .domain import Domain
from .result import Result

__all__ = ['Domain', 'Result', 'SearchSpace']
