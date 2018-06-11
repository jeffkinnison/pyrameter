from .backend_factory import backend_factory
from .local import JsonStorage
from .mongo import MongoStorage

__all__ = ['JsonStorage', 'MongoStorage']
