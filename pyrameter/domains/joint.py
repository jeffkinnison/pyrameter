"""Joint hierarchical hyperparameter domain for composition.

Classes
-------
JointDomain
    Joint hierarchical hyperparameter domain.
"""

from pyrameter.domains.base import Domain


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

    def __init__(self, *args, **kwargs):
        if len(args) > 0:
            super(JointDomain, self).__init__(args[0])
        self.domain = kwargs

    def __getattr__(self, key):
        try:
            return self.domains[key]
        except KeyError:
            return getattr(super(JointDomain, self), key)

    @property
    def complexity(self):
        if self._complexity is None:
            self._complexity = 0
            for key, val in self.domains.items():
                self._complexity += val.complexity
        return self._complexity

    def generate(self):
        return {key: val.generate() for key, val in self.domains.items()}

    def map_to_domain(self):
        pass

    def to_index(self):
        pass

    def to_json(self):
        pass
