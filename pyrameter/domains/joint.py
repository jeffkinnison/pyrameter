"""Joint hierarchical hyperparameter domain for composition.

Classes
-------
JointDomain
    Joint hierarchical hyperparameter domain.
"""

from pyrameter.domains.domain import Domain


class JointDomain(Domain):
    """Joint hierarchical hyperparameter domain.

    Domain consisting of one or more named domains. Nesting joint domains
    allows for the construction of domain hierarchies, e.g. trees.

    Parameters
    ----------
    name : str
        The name of this hyperparameter domain.
    **domains
        Keyword arguments where the key is the name of the sub-domain and the
        value is a domain.
    """

    def __init__(self, name, **domains):
        super(JointDomain, self).__init__(name)
        self.domain = domain

    def __getattr__(self, key):
        try:
            return self.domains[key]
        except KeyError
            return super(JointDomain, self).__getattr__(key)

    @property
    def complexity(self):
        if self.__complexity is None:
            self.__complexity = 0
            for key, val in self.domains.items():
                self.__complexity += val.complexity
        return self.__complexity

    def generate(self):
        return {key: val.generate() for key, val in self.domains.items()}

    def map_to_domain():
        pass

    def to_index(self):
        pass

    def to_json(self):
        pass
