"""A hyperparameter search domain.

Classes
-------
Domain
    Data model of a hyperparameter search domain.
"""

from pyrameter.db.entities.entity import Entity


class Domain(Entity):
    """Data model of a hyperparameter search domain.

    Attributes
    ----------
    domain : list or dict
        Values in the hyperparameter domain.
    path : str
        The path to this domain in the hyperparameter tree.
    complexity : float
        The size of this hyperparameter domain.

    """

    def __init__(self, domain=None, path=None):
        super(Domain, self).__init__()
        self.domain = domain
        self.path = path
        self.__complexity = None
        self.values = []

    def __call__(self, index=False):
        val = self.generate()
        self.values.append(val)
        return val if not index else (val, self.map_to_index(val))

    def __iter__(self):
        return self

    def __next__(self):
        return self.generate()

    def next(self):
        return self.__next__()

    def __eq__(self, other):
        return self.to_json() == other.to_json()

    def __lt__(self, other):
        return self.path < other.path

    def __str__(self):
        return str(self.to_json())

    @property
    def complexity(self):
        """Approximate the size of this domain.

        The size of a continuous domain is approximated by computing the
        magnitude of the interval containing 99%% of the distribution.

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
        raise NotImplementedError

    def generate(self):
        """Draw a value from this hyperparameter domain."""
        raise NotImplementedError

    def map_to_domain(self, idx, bound=False):
        """Map a index to its value in the domain.

        Parameters
        ----------
        idx : int
            The index to retrieve in the domain.
        bound : bool
            If True, return the first or last element of the domain if ``idx``
            < 0 or idx > |domain|, respectively.

        Returns
        -------
        The value at ``idx`` in the domain if the domain is discrete, else
        return the index.
        """
        return idx

    def map_to_index(self, value):
        """Map a value to its index in the domain.

        Parameters
        ----------
        value
            The value to find in the domain.

        Returns
        -------
        The index of ``value`` in the domain if the domain is discrete, else
        return the value.
        """
        return value

    def to_json(self):
        """Serialize this entity as JSON.

        Returns
        -------
        serialized : dict
            A JSON-serializable dictionary representation of this entity.
        """
        return {
            'domain': self.domain,
            'path': self.path,
            'complexity': self.__complexity
        }
