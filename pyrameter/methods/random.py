"""Generate hyperparameters purely randomly.

Functions
---------
random_search
    Randomly draw a set of hyperparameters from a search space.
"""


def random_search(space):
    """Randomly draw a set of hyperparameters from a search space.

    Parameters
    ----------
    space : list of pyrameter.domains.Domain
        The domains to draw values from.

    Returns
    -------
    values : list
        Values generated from ``space``.
    """
    return space.generate()
