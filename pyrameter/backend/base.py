"""Base backend storage engine.

Classes
-------
BaseBackend
    Abstract base class for backend storage engines.
"""


class BaseBackend(object):
    """Abstract base class for backend storage engines."""

    def load(self):
        """Load a hyperparameter search state."""
        raise NotImplementedError

    def save(self):
        """Save a hyperparameter search state."""
        raise NotImplementedError
