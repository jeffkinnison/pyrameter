from pyrameter.db import backend_factory
from pyrameter.modelgroup import ModelGroup
from pyrameter.scope import Scope


def build(specification, db=None, method='random', *args, **kwargs):
    """Construct hierarchical hyperparameter search spaces.

    Parameters
    ----------
    specification : dictionary or `pyrameter.Scope`
        The specification of the hyperparameter search space.
    db : str, optional
        Path to the database that will store search information and results.
    method : {"random","tpe","gp"}
        The hyperparameter generation strategy to use.

    Returns
    -------
    models : `pyrameter.ModelGroup`
        The collection of models in this hyperparameter search, coupled with
        the specified database backend.

    See Also
    --------
    `pyrameter.ModelGroup`
    `pyrameter.db.backend_factory`
    `pyrameter.models.model_factory`
    """
    if isinstance(specification, dict):
        specification = Scope(**specification)
    specification.model = method
    models = specification.split()
    backend = backend_factory(db, *args, **kwargs)
    model_group = ModelGroup(models=models, backend=backend)
    return model_group
