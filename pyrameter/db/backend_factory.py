import os

from pyrameter.db.base import BaseStorage


class InvalidDBPathError(Exception):
    def __init__(self, path):
        msg = 'Cannot access database at path {}.'
        super(InvalidDBPathError, self).__init__(msg.format(path))


def backend_factory(path, *args, **kwargs):
    """Create a database backend interface.

    Parameters
    ----------
    path : str
        The URL of the database. For DBMS systems, supply the full URL with
        protocol (e.g. "mongodb://<user>:<password>@<ip>:<port>"). For local
        JSON file storage, supply the path to the file or directory to save
        data to.

    Other Parameters
    ----------------
    *args
        Arguments to the backend storage object.
    **kwargs
        Keyword arguments to the backend storage object.

    Returns
    -------
    backend : {`pyrameter.db.JsonStorage`,`pyrameter.db.MongoStorage`}
        The requested backend storage object.

    Raises
    ------
    InvalidDBPathError
        Raised when the provided path does not lead to valid db.

    See Also
    --------
    `pyrameter.db.local.LocalStorage`
    `pyrameter.db.mongo.MongoStorage`
    """
    if isinstance(path, BaseStorage):
        return path
    try:
        if path.find('mongodb://') == 0:
            from pyrameter.db.mongo import MongoStorage
            return MongoStorage(path, *args, **kwargs)
        else:
            from pyrameter.db.local import JsonStorage
            return JsonStorage(path, *args, **kwargs)
    except (TypeError, AttributeError):
        raise InvalidDBPathError(path)
