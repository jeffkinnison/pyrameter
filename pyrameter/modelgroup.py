import os

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

        models = [models] if not isinstance(models, list) else models
        for model in models:
            self.add_model(model)

        self.backend = backend_factory(backend)

    def __contains__(self, id):
        return id in self.models

    def __eq__(self, other):
        return all(
            any(
                [[i == j for _, j in other.models.items()]
                    for _, i in self.models.items()]
            )
        )

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
            self.model_ids.append(model.id)
            self.models[model.id] = model
        else:
            msg = '{} is not an instance of pyrameter.models.Model'
            raise TypeError(msg.format(model))

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
            self.model_ids.pop(model_id)
        except KeyError, IndexError:
            model = None
        return None

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
            idx = np.choice(np.arange(len(self.models)), p=p)
            params = (self.model_ids[idx],) + \
                self.models[model_ids[idx]].generate()
        else:
            try:
                params = (model_id,) + self.models[model_id].generate()
            except KeyError:
                params = (None, {})
        return params

    def register_result(model_id, result_id, loss, results=None):
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
            self.models[model_id].register_result(loss, results=results)
        else:
            msg = 'No model found with id {}'.format(model_id)
            raise KeyError(msg)

    def save(self):
        self.backend.save([self.models[m] for m in self.model_ids])
