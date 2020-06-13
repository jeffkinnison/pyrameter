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

    See Also
    --------
    `pyrameter.domains.base.Domain`
    """

    def __init__(self, *args, **kwargs):
        if len(args) >= 2:
            super(ConstantDomain, self).__init__(args[0])
            self.domain = args[1]
        elif len(args) == 1:
            super(ConstantDomain, self).__init__()
            self.domain = args[0]
        else:
            raise ValueError('No domain provided.')

    @classmethod
    def from_json(cls, obj):
        """Create a new domain from a JSON encoded object.

        Parameters
        ----------
        obj : dict
            JSON object created with ``to_json``.
        
        Returns
        -------
        domain : `pyrameter.domains.exhaustive.ExhaustiveDomain`
            The domain encoded in ``obj``
        """
        domain = cls(obj['name'], obj['domain'])
        return domain

    def generate(self):
        """Generate a hyperparameter value from this domain."""
        return self.domain

    def map_to_domain(self, idx):
        """Convert an index to its value within the domain.

        This domain has a single value, so returns that value.

        Parameters
        ----------
        index : int
            Index into a discrete/categorical domain (e.g., a list).
        bound : bool, optional
            If True and ``index`` is out of bounds, return the first or last
            entry in the domain (whichever is closer). Otherwise, raises an
            IndexError if ``index`` is out of bounds.

        Returns
        -------
        value
            The value at ``index`` in the domain.

        Raises
        ------
        IndexError
            Raised when ``index`` is out of bounds and ``bound`` is ``False``.
        """
        return self.domain

    def to_index(self, value):
        """Convert a value to its index in the domain.

        This domain has a single value, so the index is always zero.

        Parameters
        ----------
        
        """
        return 0

    def to_json(self):
        """Convert the domain to a JSON-compatible format."""
        jsonified = super(ConstantDomain, self).to_json()
        jsonified.update({'domain': self.domain})
        return jsonified
