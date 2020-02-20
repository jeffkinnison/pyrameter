"""Representation of a hyperparameter domain.

Classes
-------
Domain
    Base class for hyperparameter domains.
"""

import importlib
import inspect
import itertools
import os
import re


class MetaDomain(type):
    """Metaclass for behind the scenes processes for domains."""

    def __new__(cls, name, bases, dct):
        x = super().__new__(cls, name, bases, dct)
        x._counter = itertools.count(0)
        return x


class Domain(object, metaclass=MetaDomain):
    """Base class for hyperparameter domains.

    Parameters
    ----------
    name : str
        The name of this hyperparameter domain.

    """

    def __init__(self, name=None):
        self.id = next(self.__class__._counter)
        self.name = name
        self.current = None
        self._complexity = None

    def __call__(self, *args, **kwargs):
        margs, mvargs, mkwargs, _ = inspect.getargspec(self.generate)
        if len(margs) > 1 or mvargs is not None or mkwargs is not None:
            self.current = self.generate(*args, **kwargs)
        else:
            self.current = self.generate()
        return self.current

    def __eq__(self, other):
        return self.name == other.name

    def __ge__(self, other):
        return self.name >= other.name

    def __gt__(self, other):
        return self.name > other.name

    def __hash__(self):
        return hash(self.name)

    def __le__(self, other):
        return self.name <= other.name

    def __lt__(self, other):
        return self.name < other.name

    def __ne__(self, other):
        return self.name != other.name

    @property
    def complexity(self):
        if self._complexity is None:
            self._complexity = 1
        return self._complexity

    @staticmethod
    def from_json(obj):
        mod, cls = os.path.splitext(obj['type'])
        mod = importlib.import_module(mod)
        cls = getattr(mod, cls.strip('.'))
        return cls.from_json(obj)

    def generate(self):
        """Generate a hyperparameter value from this domain."""
        raise NotImplementedError

    def map_to_domain(self, value, bound=True):
        return value

    def to_index(self, value):
        """Convert a value to its index in the domain."""
        return value

    def to_json(self):
        """Convert the domain to a JSON-compatible format."""
        classname = re.match(r"^<class '(.+)'>$",
                             str(self.__class__)).groups()[0]
        return {'name': self.name,
                'type': classname}
