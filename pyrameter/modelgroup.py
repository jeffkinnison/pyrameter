import os

import numpy as np
import scipy.stats

from pyrameter.models.model import Model
from pyrameter.db import backend_factory


class ModelGroup(object):
    """Collection of models in a hyperparameter search.

    Parameters
    ----------
    models : list of `pyrameter.models.Model`, optional
        The models to include in this group.
    complexity_sort : bool
        If true, sort models in this group by complexity.
    priority_sort : bool
        If true, sort models in this group by priority.

    Attributes
    ----------
    models : dict of `pyrameter.models.Model`
        The models in this group, indexed by model id.
    model_ids : list of str
        The ids of the models in this group. Used for sorting and selecting
        models during hyperparameter generation.
    """
    def __init__(self, models=None, backend=None, complexity_sort=True,
                 priority_sort=True):
        self.models = {}
        self.model_ids = []
        self.complexity_sort = complexity_sort
        self.priority_sort = priority_sort

        if models is not None:
            models = [models] if not isinstance(models, list) else models
            for model in models:
                self.add_model(model)

        self.backend = backend_factory(backend) \
            if backend is not None else None

    def __contains__(self, id):
        return id in self.models

    def __getitem__(self, model_id):
        return self.models[model_id]

    def __eq__(self, other):
        return all(
            any(
                [[i == j for _, j in other.models.items()]
                    for _, i in self.models.items()]
            )
        )

    def __len__(self):
        return len(self.model_ids)

    def __str__(self):
        s = '\n'.join([str(self.models[m]) for m in self.model_ids])
        s = '\n'.join(['[', s, ']'])
        return s

    def add_model(self, model):
        """Add a model to this group.

        Parameters
        ----------
        model : `pyrameter.models.model.Model`
            The model to add.

        Raises
        ------
        TypeError
            Raised if ``model`` is not an instance of `pyrameter.models.Model`
        """
        if isinstance(model, Model):
            if not self.priority_sort:
                model.priority_update_freq = -1
            self.model_ids.append(model.id)
            self.models[model.id] = model
        else:
            msg = '{} is not an instance of pyrameter.models.Model'
            raise TypeError(msg.format(model))

    def clear(self):
        """Clear this model group of all models."""
        self.models = {}
        self.model_ids = []

    def remove_model(self, model_id):
        """Pop a model from the group.

        Parameters
        ----------
        model_id : str
            The id of the model to pop.

        Returns
        -------
        model
            The popped model, or None if a model with the requested id is not
            in this group.
        """
        try:
            model = self.models.pop(model_id)
            self.model_ids.remove(model_id)
        except (KeyError, IndexError):
            model = None
        return model

    def sort_models(self):
        """Sort models by their complexity/priority rank.

        Sorts the list of models by their rank, a linear combination of
        individual model complexity and priority.

        Notes
        -----
        The complexity and priority heuristics are used as defined by
        Kinnison *et al*[1]_

        References
        ---------
        .. _[1]
        """
        for v in self.models.values():
            v.rank = 1

        if self.complexity_sort:
            self.model_ids.sort(key=lambda m: self.models[m].complexity,
                                reverse=True)
            for i in range(len(self.model_ids)):
                self.models[i].rank *= i

        if self.priority_sort:
            self.model_ids.sort(key=lambda m: self.models[m].priority,
                                reverse=True)
            for i in range(len(self.model_ids)):
                self.models[i].rank *= i

        self.model_ids.sort(key=lambda m: self.models[m].rank, reverse=True)

    def generate(self, model_id=None):
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
        if not model_id:
            if self.complexity_sort or self.priority_sort:
                p = np.array([scipy.stats.planck.pmf(i, 0.5)
                             for i in range(len(self.models))])
            else:
                p = np.ones(len(self.models))
            p = p / p.sum()
            idx = np.random.choice(np.arange(len(self.models)), p=p)
            params = (self.model_ids[idx],) + \
                self.models[self.model_ids[idx]]()
        else:
            try:
                params = (model_id,) + self.models[model_id]()
            except KeyError:
                params = (None, {})
        return params

    def optimal(self, mode='best', count=1):
        """Get the optimal observed result(s) from among the models.

        Parameters
        ----------
        mode : {'best','model'}
            The format of the results. If "best", return the best observed
            result from across all models. If "model", return the best observed
            result for each model.
        count : int, optional
            The number of results to return. If mode is "model", return this
            many results for each model. Default 1.

        Returns
        -------
        optimal : dict
            A dictionary of lists of results indexed by model id.
        """
        if mode == 'model':
            results = self._optimal_model_mode(count)
        else:
            results = self._optimal_best_mode(count)
        return results

    def _optimal_best_mode(self, count):
        """Get the optimal observed results across all models.

         Parameters
         ----------
         count : int
             The number of results to return. If mode is "model", return this
             many results for each model.

        Returns
        -------
        optimal : dict
            A dictionary of lists of results indexed by model id.
        """
        optimal = []
        results = []

        # Concatenate the results from each model into a single list
        for mid in self.model_ids:
            model = self.models[mid]
            results.extend(
                [r.to_json() for r in model.results if r.loss is not None])

        # Sort the results and extract the ``count`` best.
        results.sort(key=lambda x: x['loss'])
        results = {r['model']: r for r in results[:count]}
        return results

    def _optimal_model_mode(self, count):
        """Get the optimal observed results from each model.

         Parameters
         ----------
         count : int
             The number of results to return. If mode is "model", return this
             many results for each model.

        Returns
        -------
        optimal : dict
            A dictionary of lists of results indexed by model id.
        """
        optimal = {}
        for mid in self.model_ids:
            model = self.models[mid]
            results = sorted(
                [r.to_json() for r in model.results if r.loss is not None],
                key=lambda x: x['loss'])
            optimal[mid] = results[:count]
        return optimal

    def register_result(self, model_id, result_id, loss, results=None):
        """Add a result to the given model.

        Parameters
        ----------
        model_id : str
            The model that the result belongs to.
        result_id : str
            The id of this result if available.
        loss : float
            The loss value associated with this result.
        results : dict, optional
            Additional values to store.
        """
        if model_id in self.models:
            submissions, params = \
                self.models[model_id].register_result(result_id,
                                                      loss,
                                                      results=results)
        else:
            msg = 'No model found with id {}'.format(model_id)
            raise KeyError(msg)

        return submissions, params

    @property
    def result_count(self):
        return sum([len(m.results) for m in self.models.values()])

    def save(self):
        self.backend.save([self.models[m] for m in self.model_ids])
