"""A domain containing discrete or categorical values.

Classes
-------
DiscreteDomain
    Hyperparameter defined over a categorical domain.

"""

import numpy as np
from scipy.stats import randint

from pyrameter.db.entities.domain import Domain


class DiscreteDomain(Domain):
    """Hyperparameter defined over a discrete domain.

    Discrete domains are sets of arbitrary, non-overlapping values that have meaning
    to what they parameterize. Values are drawn randomly when generated.

    Parameters
    ----------
    domain : object or list of object
        The set of objects comprising this domain.
    path : str
        Path to this domain in the search hierarchy.

    Notes
    -----
    If a single, non-list object is provided to a DiscreteDomain, it will be
    wrapped in a list to represent a domain with a single value.
    """
    def __init__(self, domain, path='', random_state=None):
        if random_state:
            self.random_state = random_state
        else:
            self.random_state = \
                np.random.RandomState(np.random.randint(2147483647))
        super(DiscreteDomain, self).__init__(domain, path=path)

    @property
    def complexity(self):
        """Approximate the size of this domain.

        The size of a discrete domain is approximated by computing the
        cardinality of the domain.

        Notes
        -----
        This property implements the complexity formula for discrete domains
        introduced by Kinnison *et al.* _[1]

        References
        ----------
        ..  [1] Kinnison, J., Kremer-Herman, N., Thain, D., & Scheirer, W.
            (2017). SHADHO: Massively Scalable Hardware-Aware Distributed
            Hyperparameter Optimization. arXiv preprint arXiv:1707.01428.
        """
        if self.__complexity is None:
            self.__complexity = 2.0 - (1.0 / len(self.domain))
        return self.__complexity

    def generate(self):
        """Generate a value from this domain.

        Parameters
        ----------
        index : bool
            If True, return the index of the generated value along with the
            value.

        Returns
        -------
        value
            A value drawn from this domain.
        index : int, optional
            The index of ``value`` in this domain.
        """
        idx = randint.rvs(0, len(self.domain), random_state=self.random_state)
        value = self.domain[idx]
        return value

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
        if bound:
            idx = int(round(idx))
            idx = min(len(self.domain) - 1, max(0, idx))
        try:
            val = self.domain[idx]
        except IndexError:
            val = None
        return val

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
        try:
            idx = self.domain.index(value)
        except ValueError:
            idx = None
        return idx

    def to_json(self):
        """Convert this domain into a JSON-serializable format.

        Returns
        -------
        domain : dict
            A dictionary representation of this domain containing only valid
            JSON values.
        """
        j = super(DiscreteDomain, self).to_json()
        return j
