"""A hyperparameter search domain.

Classes
-------
Domain

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

    """

    def __init__(self, domain=None, path=None):
        super(Domain, self).__init__()
        self.domain = domain
        self.path = path

    def to_json(self):
        """Serialize this entity as JSON.

        Returns
        -------
        serialized : dict
            A JSON-serializable dictionary representation of this entity.
        """
        return {
            'domain': self.domain,
            'path': self.path
        }
