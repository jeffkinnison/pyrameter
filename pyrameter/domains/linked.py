"""Hyperparameter domain that is dependent upon another domain's output.

Classes
-------
LinkedDomain
    Hyperparameter domain dependent upon a different domain.

"""

from pyrameter.db.entities import Domain


class LinkedDomain(Domain):
    """Hyperparameter domain dependent upon a different domain.

    In many cases, a hyperparameter value is dependent upon another
    hyperparameter value, for example computing the padding of a convolutional
    layer or inferring the shape of an input tensor. The LinkedDomain
    references the most recent generated value of another hyperparmeter domain
    and optionally applies a callback.

    Parameters
    ----------
    domain : pyrameter.Domain
        The domain to link to this domain.
    path : str
        The unique identifier of this domain.
    callback : function
        Function applied to modify the linked value when generated, e.g.
        computing 'same' padding for a convolutional layer with shape k as
        `floor(k / 2)`
    """

    def __init__(self, domain, path='', callback=None):
        self.callback = callback if callback is not None else (lambda x: x)

        if not isinstance(domain, Domain):
            raise TypeError(
                "`domain` argument must be an instance of pyrameter.Domain")

        super(LinkedDomain, self).__init__(domain, path=path)

    def __lt__(self, other):
        """Ensure that this domain is sorted after the domain it links to."""
        return not self.domain == other

    @property
    def complexity(self):
        return 1

    def generate(self):
        """Return the most recently-generated value of the linked domain."""
        val = self.domain.values[-1]
        val = self.callback(val)
        return val
