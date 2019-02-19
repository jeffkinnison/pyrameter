"""A hyperparameter search domain containing a single value.

Classes
-------
ConstantDomain
    Data model of a single-member hyperparameter domain.

"""

from pyrameter.db.entities.domain import Domain


class ConstantDomain(Domain):
    """Data model of a single-member hyperparameter domain.

    Attributes
    ----------
    domain
        Value of the hyperparameter domain.
    path : str
        The path to this domain in the hyperparameter tree.
    complexity : float
        The size of this hyperparameter domain.

    """

    def __init__(self, domain=None, path=None):
        super(ConstantDomain, self).__init__(domain=domain, path=path)

    @property
    def complexity(self):
        """Approximate size of this domain.

        The size of a continuous domain is approximated by computing the
        magnitude of the interval containing 99%% of the distribution.

        Returns
        -------
        complexity : float

        Notes
        -----
        This property implements the complexity formula for continuous domains
        introduced by Kinnison *et al.* _[1]

        References
        ----------
        ..  [1] Kinnison, J., Kremer-Herman, N., Thain, D., & Scheirer, W.
            (2017). SHADHO: Massively Scalable Hardware-Aware Distributed
            Hyperparameter Optimization. arXiv preprint arXiv:1707.01428.
        """
        if self.__complexity is None:
            self.__complexity = 1.0
        return self.__complexity

    def generate(self):
        return self.domain
