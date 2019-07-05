"""Data model for a hyperparameter trial.

Classes
-------
Trial
    Presistent instance of a hyperparameter trial.
"""

import enum
import itertools


@enum.unique
class TrialState(enum.Enum):
    """Representation of discrete trial states.

    Attributes
    ----------
    INIT
        Trial object has been initialized but provided no parameters.
    READY
        Parameters have been provided to the trial.
    ERROR
        An error occurred while running the trial.
    DONE
        The trial completed with no errors and recorded a result.
    """
    INIT = enum.auto()
    READY = enum.auto()
    ERROR = enum.auto()
    DONE = enum.auto()


class TrialMeta(type):
    """Metaclass for handling behind-the-scenes tasks for Trial objects."""
    def __new__(cls, name, bases, dct):
        x = super().__new__(cls, name, bases, dct)
        x._counter = itertools.count(0)


class Trial(object, metaclass=TrialMeta):
    """Persistent instance of a hyperparameter evaluation trial.

    Parameters
    ----------
    parameters : dict
        Hyperparameter values to be used in this trial.
    results : dict
        Results of evaluating a model trained with ``parameters``, e.g. loss,
        accuracy, running time, etc.
    objective : float
        The value to optimize on.
    errmsg : str
        Information about errors encountered in this trial.

    Attributes
    ----------
    id : int or bson.objectid.ObjectId

    """

    def __init__(self, searchspace, parameters=None, results=None,
                 objective=None, errmsg=None):
        self.id = next(self.__class__._counter)
        self.status = None
        self.dirty = False

        self.searchspace = searchspace
        self.parameters = parameters
        self.results = results
        self.objective = objective
        self.errmsg = errmsg

        self.set_status()

    def __setattr__(self, key, val):
        start_val = getattr(self, key)
        super().__setattr__(key, val)
        if start_val != val:
            self.dirty = True
            self.set_status()

    def set_status(self):
        """Introspect to set the state of this trial."""
        if self.errmsg is not None:
            self.status = TrialState.ERROR
        elif self.results is not None and self.parameters is not None:
            self.status = TrialState.DONE
        elif self.parameters is not None:
            self.status = TrialState.READY
        else:
            self.status = TrialState.INIT

    def to_json(self):
        """Convert this trial to a JSON-compatible representation."""
        return dict(
            searchspace=self.searchspace.id if hasattr(self.searchspace, 'id')
                        else self.searchspace,
            status=self.status.value,
            hyperparameters=self.parameters,
            results=self.results,
            objective=self.objective
        )
