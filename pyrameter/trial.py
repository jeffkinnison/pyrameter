"""Data model for a hyperparameter trial.

Classes
-------
Trial
    Presistent instance of a hyperparameter trial.
"""

import enum
import itertools
import re
import weakref


@enum.unique
class TrialStatus(enum.Enum):
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
        return x


class Trial(object, metaclass=TrialMeta):
    """Persistent instance of a hyperparameter evaluation trial.

    Parameters
    ----------
    hyperparameters : list
        Hyperparameter values to be used in this trial.
    domains : list of pyrameter.Domain
        The domains that generated ``hyperparameters`` in order.
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
        Unique id of this trial.
    status : pyrameter.trials.TrialStatus

    """

    def __init__(self, searchspace, hyperparameters=None, results=None,
                 objective=None, errmsg=None):
        self.id = next(self.__class__._counter)
        self.dirty = False

        self._searchspace = weakref.ref(searchspace) \
                           if searchspace is not None else None
        self.hyperparameters = hyperparameters
        self.results = results
        self.objective = objective
        self.errmsg = errmsg

        self.submissions = 0

        self.status = None
        self.set_status()

    def __hash__(self):
        return hash(str(self.id))

    def __setattr__(self, key, val):
        if key in self.__dict__:
            start_val = self.__dict__[key]
        else:
            start_val = None
        self.__dict__[key] = val
        if 'status' in self.__dict__ and start_val != val:
            self.set_status()

    def __eq__(self, other):
        return (self.searchspace == other.searchspace and
                self.hyperparameters == other.hyperparameters and
                self.results == other.results and
                self.objective == other.objective and
                self.errmsg == other.errmsg and
                self.status == other.status)
    
    def flatten_results(self):
        if self.results is None:
            return {}

        flat = {}

        def recurse_nested(current, name=''):
            if not isinstance(current, dict):
                if name == '':
                    name = 'results'
                flat[name] = current
            else:
                for key in current.keys():
                    recurse_nested(current[key], name='.'.join([name, key]))
        
        recurse_nested(self.results)
        return flat


    @classmethod
    def from_json(cls, obj):
        trial = cls(obj['searchspace'],
                    hyperparameters=obj['hyperparameters'],
                    results=obj['results'],
                    objective=obj['objective'],
                    errmsg=obj['errmsg'])
        trial.dirty = False
        return trial

    @property
    def parameter_dict(self):
        """Convert the trial hyperparameters into a nested dictionary.

        Hyperparameters are in general defined as hierarchies. Ensuring that
        generated hyperparameter values are available in the same hierarchy as
        they were defined provides a consistent interface for users.

        Returns
        -------
        hyperparameters : dict
            The (potentially nested) dictionary of hyperparameters values used
            in this trial, structured to match the original hyperparameter
            specification.
        """
        params = {}
        for i, domain in enumerate(self.searchspace.domains):
            curr = params
            path = domain.name.strip('.').split('.')
            for p in path:
                if p not in curr:
                    if re.search(r'[_][_][_][\d]+', p):
                        pnew, num = p.split('___')
                        num = int(num)
                        curr[pnew] = []
                    else:
                        curr[p] = {}

                if re.search(r'[_][_][_][\d]+', p):
                    pnew, num = p.split('___')
                    num = int(num)
                    if len(curr[pnew]) <= num:
                        for i in range(len(curr[pnew]), num + 1):
                            curr[pnew].append(None)

                    if p != path[-1]:
                        curr[pnew][num] = {}
                        curr = curr[pnew][num]
                    else:
                        curr[pnew][num] = self.hyperparameters[i]
                else:
                    if p != path[-1]:
                        curr = curr[p]
                    else:
                        curr[p] = self.hyperparameters[i]
        return params

    @property
    def searchspace(self):
        return self._searchspace()

    def set_status(self):
        """Introspect to set the state of this trial."""
        if 'status' in self.__dict__:
            oldstatus = self.status
            if self.errmsg is not None:
                self.status = TrialStatus.ERROR
            elif self.results is not None and self.hyperparameters is not None:
                self.status = TrialStatus.DONE
            elif self.hyperparameters is not None:
                self.status = TrialStatus.READY
            else:
                self.status = TrialStatus.INIT
            self.dirty = oldstatus == self.status

    def to_json(self):
        """Convert this trial to a JSON-compatible representation."""
        return dict(
            searchspace=self.searchspace.id
                        if hasattr(self.searchspace, 'id')
                        else self.searchspace,
            status=self.status.value,
            hyperparameters=self.hyperparameters,
            results=self.results,
            objective=self.objective,
            errmsg=self.errmsg
        )
