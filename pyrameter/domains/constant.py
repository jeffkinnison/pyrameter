"""Representation of a singleton hyperparameter domain.

Classes
-------
ConstantDomain
    A singleton hyperparameter domain.
"""

from pyrameter.domains.base import Domain


class ConstantDomain(Domain):
    """A singleton hyperparameter domain.

    Parameters
    ----------
    name : str
        The name of this hyperparameter domain.
    domain
        The single value in this domain.
    """

    def __init__(self, name, domain):
        super(ConstantDomain, self).__init__(name)
        self.domain = domain

    def generate(self):
        """Generate a hyperparameter value from this domain."""
        return self.domain
