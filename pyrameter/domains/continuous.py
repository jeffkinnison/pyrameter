"""A continuous domain defined by a probability distribution.

Classes
-------
ContinuousDomain
    Hyperparameter defined over a continuous, real-valued domain.

"""

import numpy as np
import scipy.stats

from tesserae.db.entities.domain import Domain


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
        if not isinstance(domain, scipy.stats.rv_continuous):
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
            lo, hi = self.domain.interval(.99, *self.domain_args,
                                          **self.domain_kwargs)
            self.__complexity = 2.0 + np.linalg.norm(hi - lo)
        return self.__complexity

    def generate(self, index=False):
        """Draw a value from this domain.

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
        j['domain'] = {
            'distribution': self.domain.name,
            'args': self.domain_args,
            'kws': self.domain_kwargs}
        return j
