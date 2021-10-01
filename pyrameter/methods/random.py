"""Generate hyperparameters purely randomly.

Classes
---------
RandomSearch
    Randomly draw a set of hyperparameters from a search space.
"""

from pyrameter.methods.method import Method


class RandomSearch(Method):
    """Randomly draw a set of hyperparameters from a search space.
    
    Notes
    -----
    The implementation here is somewhat different from other methods in that
    the search space is directly used to generate values. For reference
    implementations, see pretty much any other method in `pyrameter.methods`.
    """

    def generate(self, trial_data, domains):
        """Randomly generate a set of hyperparameters.

        Parameters
        ----------
        space : pyrameter.searchspace.SearchSpace
            The domains to draw values from.

        Returns
        -------
        values : list
            Values generated from ``space``.
        """
        return [d.generate() for d in domains]
