"""Representation of a discrete hyperparameter domain.

Classes
-------
DiscreteDomain
    A discrete hyperparameter domain.
"""

import dill
import numpy as np
import scipy.stats

from pyrameter.domains.base import Domain


class DiscreteDomain(Domain):
    """A Discrete hyperparameter domain.

    Parameters
    ----------
    name : str
        The name of this hyperparameter domain.
    domain : list
        A collection of values in the domain.
    callback : callable, optional
        An optional callback to run on generated hyperparameter values, e.g. to
        scale or otherwise modify the value.
    seed : int or numpy.random.RandomState, optional
        The random seed or random state to use to generate values.

    """

    def __init__(self, *args, **kwargs):
        if len(args) >= 2:
            super(DiscreteDomain, self).__init__(args[0])
            domain = args[1]
        elif len(args) == 1:
            super(DiscreteDomain, self).__init__()
            domain = args[0]
        else:
            raise ValueError('No domain provided.')

        callback = kwargs.pop('callback', None)

        if not isinstance(domain, list) or isinstance(domain, (str, tuple)):
            domain = [domain]
        elif isinstance(domain, range):
            domain = list(domain)

        self.domain = list(domain)

        self.callback = callback if callback is not None else lambda x: x

    def bound_index(self, idx):
        """Clamp an index into the domain to its viable values.

        Parameters
        ----------
        idx : int
            The index to clamp.

        Returns
        -------
        idx : int
            The index clamped to the range ``[0, n_entries]``.
        """
        return int(min(max(0, idx), len(self.domain)))

    @property
    def bounds(self):
        """The viable lower and upper bounds of the domain.

        For discrete domains, returns the first and last index of the domain,
        always ``(0, n_elements)``.

        Returns
        -------
        low, high : float
            The lower and upper bounds of the domain.
        """
        return (0, len(self.domain))

    @property
    def complexity(self):
        if self._complexity is None:
            try:
                self._complexity = 2 - (1 / len(self.domain))
            except ZeroDivisionError:
                self._complexity = 1
        return self._complexity

    @classmethod
    def from_json(cls, obj):
        if 'random_state' in obj:
            rng = obj['random_state']
            random_state = np.random.RandomState()
            random_state.set_state((rng[0], np.array(rng[1], dtype=np.uint32),
                                    rng[2], rng[3], rng[4]))
            del obj['random_state']
        else:
            random_state = obj['random_state']
            del obj['random_state']

        try:
            callback = dill.loads(obj['callback'])
        except KeyError:
            callback = None
        
        domain = cls(obj['name'], obj['domain'],
                     callback=callback, seed=random_state)
        
        domain.id = obj['id']
        domain.current = obj['current']
        
        return domain

    def generate(self):
        """Generate a hyperparameter value from this domain."""
        if len(self.domain) > 0:
            index =  scipy.stats.randint.rvs(0, len(self.domain),
                                        random_state=self._rng.rng)
            return index
        else:
            return None

    def map_to_domain(self, idx, bound=True):
        if bound:
            idx = int(round(idx))
            idx = min(len(self.domain) - 1, max(0, idx))
        elif not bound and idx < 0:
            idx = len(self.domain)
        try:
            val = self.domain[idx]
        except IndexError:
            val = None
        return val

    def to_index(self, value):
        """Convert a value to its index in the domain."""
        try:
            idx = self.domain.index(value)
        except ValueError:
            idx = None
        return idx

    def to_json(self):
        jsonified = super(DiscreteDomain, self).to_json()
        jsonified.update({
            'domain': list(self.domain)
        })
        if isinstance(self.random_state, np.random.RandomState):
            rs = self.random_state.get_state()
            jsonified.update({
                'random_state': rs
            })
        else:
            jsonified.update({'random_state': self.random_state})
        return jsonified
