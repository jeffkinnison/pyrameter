"""A common interface for pyrameter database entities.

Classes
-------
Entity
    Base class for pyrameter entities for a standard interface.

"""

from itertools import count

from six import with_metaclass


class EntityMeta(type):
    """Metaclass for counting instances of specific entity classes."""
    def __new__(cls, name, bases, attrs):
        cls = super(EntityMeta, cls).__new__(cls, name, bases, attrs)
        cls.__counter = count(0)
        cls.__dirty = set()
        return cls


class Entity(with_metaclass(EntityMeta)):
    """Abstract representation of a database entity.

    This class should be subclassed to create new data models for
    hyperparameter optimization. It manages JSON deserialization, automatic
    unique id assignment, and tracking modified entities for updates.
    """

    def __init__(self):
        self.id = next(self.__class__.__counter)

    def __setattr__(self, attr, val):
        self.__class__.__dirty.add(self)
        super(Entity, self).__setattr__(attr, val)

    def to_json(self):
        """Serialize this entity as JSON.

        Raises
        ------
        NotImplementedError
            Override this method in subclasses.
        """
        raise NotImplementedError

    @classmethod
    def from_json(cls, document, mark_dirty=False):
        """Deserialize this entity from JSON.

        Parameters
        ----------
        document : dict
            Dictionary of values to populate the entity with.
        mark_dirty : bool
            If True, mark the created entity to be updated on save. If loading
            from database with no changes, set to False.

        Returns
        -------
        obj : Entity
            The database entity instance loaded from ``document``.
        """
        # Remove standard entries that would not be in the entity __init__
        try:
            obj_id = document['id']
            del document['id']
        except KeyError:
            obj_id = None

        # Create the instance.
        obj = cls(**document)

        # Set the correct ID.
        if obj_id is not None:
            super(Entity, obj).__setattr__('id', obj_id)

        # Remove from the update list if no true changes have been made.
        if not mark_dirty:
            try:
                cls.__dirty.remove(obj)
            except KeyError:
                pass

        return obj

    @classmethod
    def serialize_updated(cls):
        """Serialize all entities of this class that have been updated."""
        return [e.to_json() for e in cls.__dirty]
