import uuid

import numpy as np
import scipy.stats
from scipy.stats import randint


class Domain(object):
    """Base class for defining search domains.

    Parameters
    ----------
    domain
        The set or range of values to search.
    path : str
        Path to this domain in the search hierarchy.

    Notes
    -----
    ``path`` is automatically computed when models are created during the
    splitting process.
    """
    def __init__(self, domain=None, path='', callback=None):
        self.id = str(uuid.uuid4())
        self.domain = domain
        self.path = path
        self.callback = callback if callable(callback) else lambda x: x
        self._complexity = None

    def __getitem__(self, key):
        if hasattr(self, key):
            return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __iter__(self):
        return self

    def __next__(self):
        return self.generate()

    def next(self):
        return self.__next__()

    def __eq__(self, other):
        return self.to_json() == other.to_json()

    def __str__(self):
        return str(self.to_json())

    def generate(self, index=False):
        raise NotImplementedError

    def complexity(self):
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
        return {
            'type': self.__class__.__name__,
            'path': self.path
        }

    @staticmethod
    def from_json(spec):
        if spec['type'] == ContinuousDomain.__name__:
            return ContinuousDomain(spec['distribution'], path=spec['path'],
                                    *spec['args'], **spec['kws'])
        elif spec['type'] == DiscreteDomain.__name__:
            return DiscreteDomain(spec['domain'], path=spec['path'])
        elif spec['type'] == ExhaustiveDomain.__name__:
            return ExhaustiveDomain(spec['domain'], path=spec['path'],
                                    idx=spec['idx'])


class ContinuousDomain(Domain):
    """Hyperparameter defined over a continuous, real-valued domain.

    Parameters
    ----------
    domain : `scipy.stats.rv_continuous`
        The probability distribution defining both the domain and how values
        are drawn.
    path : str
        Path to this domain in the search hierarchy.

    Other Parameters
    ----------------
    args
        Additional arguments to parameterize ``domain``.
    kws
        Additional keyword arguments to parameterize ``domain``.
    """
    def __init__(self, domain, path='', callback=None, random_state=None,
                 *args, **kwargs):
        if not isinstance(domain, scipy.stats._continuous_distns.uniform_gen):
            domain = getattr(scipy.stats, str(domain))
        if random_state:
            self.random_state = random_state
        else:
            self.random_state = \
                np.random.RandomState(np.random.randint(2147483647))
        self.domain_args = args
        self.domain_kwargs = kwargs
        super(ContinuousDomain, self).__init__(domain,
                                               path=path,
                                               callback=callback)

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
        if self._complexity is None:
            a, b = self.domain.interval(.99, *self.domain_args,
                                        **self.domain_kwargs)
            self._complexity = 2.0 + np.linalg.norm(b - a)
        return self._complexity

    def generate(self, index=False):
        """Generate a value from this domain.

        Returns
        -------
        value : float
            A value drawn from this domain's probability distribution.
        """
        return self.callback(
            self.domain.rvs(*self.domain_args,
                            random_state=self.random_state,
                            **self.domain_kwargs))

    def to_json(self):
        """Convert this domain into a JSON-serializable format.

        Returns
        -------
        domain : dict
            A dictionary representation of this domain containing only valid
            JSON values.
        """
        j = super(ContinuousDomain, self).to_json()
        j.update({'distribution': self.domain.name,
                  'args': self.domain_args,
                  'kws': self.domain_kwargs})
        return j


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
    def __init__(self, domain, path=''):
        try:
            self.rng = randint(0, len(domain))
        except AttributeError:
            domain = [domain]
            self.rng = randint(0, len(domain))
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
        if self._complexity is None:
            self._complexity = 2.0 - (1.0 / len(self.domain))
        return self._complexity

    def generate(self, index=False):
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
        idx = self.rng.rvs()
        value = self.domain[idx]
        return value if not index else (value, idx)

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

    def map_to_index(self, val):
        """Map a value to its index in the domain.

        Parameters
        ----------
        val
            The value to find in the domain.

        Returns
        -------
        The index of ``value`` in the domain if the domain is discrete, else
        return the value.
        """
        try:
            idx = self.domain.index(val)
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
        j.update({'domain': self.domain})
        return j


class ExhaustiveDomain(Domain):
    """Hyperparameter defined over a discrete domain, searched exhaustively.

    Exhaustive domains iterate over the values they contain in order, making
    them suitable for grid search or searches where all combinations of
    hyperparameter values must be tested.

    Parameters
    ----------
    domain : object or list of object
        The set of objects comprising this domain.
    path : str
        Path to this domain in the search hierarchy.

    Notes
    -----
    If a single, non-list object is provided to an ExhaustiveDomain, it will be
    wrapped in a list to represent a domain with a single value.
    """
    def __init__(self, domain, path='', idx=0):
        self.idx = idx
        if not isinstance(domain, list):
            domain = [domain]
        super(ExhaustiveDomain, self).__init__(domain, path=path)

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
        if self._complexity is None:
            self._complexity = 2.0 - (1.0 / len(self.domain))
        return self._complexity

    def generate(self, index=False):
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
        idx = self.idx
        val = self.domain[idx]
        self.idx = (self.idx + 1) % len(self.domain)
        return val if not index else (val, idx)

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

    def map_to_index(self, val):
        """Map a value to its index in the domain.

        Parameters
        ----------
        val
            The value to find in the domain.

        Returns
        -------
        The index of ``value`` in the domain if the domain is discrete, else
        return the value.
        """
        try:
            idx = self.domain.index(val)
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
        j = super(ExhaustiveDomain, self).to_json()
        j.update({'domain': self.domain, 'idx': self.idx})
        return j
