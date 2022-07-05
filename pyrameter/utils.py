"""Utilities for managing the pyrameter experience.

Classes
-------
CountedBase
    Base class for classes that should be counted/given a unique id.
"""
import functools
import itertools
import json
import re

import numpy as np


class CountedBase(object):
    """Base class for classes that should be counted/given a unique id.

    Attributes
    ----------
    counter : itertools.count
        Counter tranking the number of class instances created.
    """

    counter = itertools.count(0)

    def __init__(self):
        self.id = next(self.__class__.counter)


class PyrameterEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.ndarray, np.generic)):
            return {
                '__data': obj.tolist(),
                '__dtype': str(obj.dtype),
            }
        else:
            return super(PyrameterEncoder, self).default(obj)


class PyrameterDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        super(PyrameterDecoder, self).__init__(
            *args, object_hook=self.object_hook, **kwargs)

    def object_hook(self, obj):
        if isinstance(obj, dict) and sorted(obj.keys()) == ['__data', '__dtype']:
            if isinstance(obj['__data'], list):
                arr = np.array(obj['__data']).astype(obj['__dtype'])
            else:
                arr = np.dtype(obj['__dtype']).type(obj['__data'])
            return arr
        else:
            return obj


def partialize(func):
    """Create partials with kwargs only for pass-through parameterzation.

    Parameters
    ----------
    func : callable
        Function to convert into a partial.
    
    Returns
    -------
    wrapper : callable
        The wrapped ``func`` which stores any keyword args passed for
        runtime currying.
    
    Notes
    -----
    This formulation exists to allow for default keyword args for
    hyperparameter generation methods and pass-through parameterization.
    This allows users to specify their generation method by name in the
    simple case.

    Examples
    --------
    >>> @partialize
    ... def add(x, y=1):
    ...     return x + y
    >>> plus_two = add(y=2)
    >>> plus_two(5)
    7
    """
    @functools.wraps(func)
    def wrapper(**kwargs):
        return functools.partial(func, **kwargs)
    return wrapper
