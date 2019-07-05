"""Hyperparameter domain for exhaustive grid search.

Classes
-------
ExhaustiveDomain
    Discrete/categorical domain for exhaustive grid search.
"""

from pyrameter.domains.base import Domain


class ExhaustiveDomain(Domain):
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
    graphs.
    """

    def __init__(self, name, domain):
        super(ExhaustiveDomain, self).__init__(name)
        self.domain = domain
