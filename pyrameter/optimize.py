"""Base class for a hyperparameter optimizer.

Classes
-------
Optimizer
    Base class for hyperparameter optimizer.

"""

import numpy as np
import scipy.stats

import pyrameter.methods as methods
from pyrameter.scope import Scope


class Optimizer(object):
    """Base class for hyperparameter optimizers.

    Parameters
    ----------
    exp_key : str
        Unique id of the experiment.
    search_spaces
        Search space definitions consisting of scopes, domains, or the
        pyrameter markup.
    backend
        The db backend to use, e.g. a JSON filepath or MongoDB URL/connection.
    method
        The hyperparameter generation method to use.

    Attributes
    ----------
    exp_key : str
        Unique id of the experiment.
    search_space_ids
        List of search space ids for random access, sorting, etc.
    search_spaces
        Search space id/object key/value store for fast lookup.
    backend
        The db backend to use, e.g. a JSON filepath or MongoDB URL/connection.
    method
        The hyperparameter generation method to use.
    """

    def __init__(self, exp_key, search_spaces, backend, method,
                 complexity_sort=True, uncertainty_sort=True, **method_kws):
        self.exp_key = exp_key

        self.search_spaces = Scope.build(search_spaces)
        self.search_space_ids = [s.id for s in self.search_spaces]

        try:
            self.method = method if not isinstance(method, str) \
                          else getattr(methods, method)(**method_kws)
        except KeyError:
            msg = 'No hyperparameter search method {} defined in ' + \
                  'pyrameter.methods. Please see the documentation for ' + \
                  'pyramter.methods for valid methods.'
            raise KeyError(msg.format(method))

        self.backend = backend
        self.complexity_sort = bool(complexity_sort)
        self.uncertainty_sort = bool(uncertainty_sort)

        self.__idx = None

    def __getitem__(self, start, stop=None):
        if not stop:
            id_slice = slice(start)
        else:
            id_slice = slice(start, stop)
        return [self.search_spaces[sid]
                for sid in self.search_space_ids[id_slice]]

    def __iter__(self):
        self.__idx = 0
        return self

    def __next__(self):
        try:
            key = self.search_space_ids[self.__idx]
            search_space = self.search_spaces[key]
            self.__idx += 1
            return search_space
        except IndexError:
            raise StopIteration

    def next(self):
        return self.__next__()

    def generate(self, search_space_id=None):
        """Generate a set of hyperparameter values from a model.
        Generate hyperparameter values for a probabilistically-selected or
        user-specified model in this group.
        Parameters
        ----------
        model_id : str, optional
            The id of the model to generate values for. If not specified, the
            model is selected probabilistically.
        Returns
        -------
        model_id
            The id of the model the hyperparameter values belong to.
        result_id
            The id of the result structure these values are associated with.
        params
            The set of hyperparameter values.
        Notes
        -----
        Probabilistic model selection follows a discrete planck distribution
        limited to the number of models in the group.
        """
        if search_space_id is None:
            if self.complexity_sort or self.uncertainty_sort:
                probs = np.array([scipy.stats.planck.pmf(i, 0.5)
                                  for i in range(len(self.search_spaces))])
            else:
                probs = np.ones(len(self.search_spaces))
            probs = probs / probs.sum()
            idx = np.random.choice(np.arange(len(self.search_spaces)), p=probs)
            params = (self.search_space_ids[idx],) + \
                self.search_spaces[self.search_space_ids[idx]]()
        else:
            try:
                params = (search_space_id,) + \
                          self.search_spaces[search_space_id]()
            except KeyError:
                params = (None, {})
        return params

    @classmethod
    def load(cls, backend):
        search_spaces = backend.load()

    def best(self, mode='global', count=1):
        pass

    def save(self):
        self.backend.save(self.search_spaces)
