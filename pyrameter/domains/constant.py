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
        domain = cls(obj['name'], obj['domain'])
        return domain

    def generate(self):
        """Generate a hyperparameter value from this domain."""
        return self.domain

    def map_to_domain(self, idx):
        return self.domain

    def to_index(self, value):
        return 0

    def to_json(self):
        """Convert the domain to a JSON-compatible format."""
        jsonified = super(ConstantDomain, self).to_json()
        jsonified.update({'domain': self.domain})
        return jsonified
