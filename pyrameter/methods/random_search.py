"""Random search for hyperparameter optimization.

Functions
---------
random_search
    Randomly generate hyperparameter values.
"""


def random_search(search_space):
    """Randomly generate hyperparameter values.

    Parameters
    ----------
    search_space : pyrameter.db.SearchSpace
        The search space to draw values from.

    Returns
    -------
    hyperparameters : dict
        The set of hyperparameters generated from this search space.
    """
    return search_space.generate()
