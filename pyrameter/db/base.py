"""Base database storage functionality.

Classes
-------
BaseStorage
    Abstract class for database storage classes to inherit from.
"""


class BaseStorage(object):
    """Abstract class for database storage classes to inherit from."""

    def load(self):
        raise NotImplementedError

    def save(self, models):
        raise NotImplementedError
