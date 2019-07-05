from pyrameter.domains import *


def build(specification, root=None):
    """Construct hierarchical hyperparameter search spaces.

    Parameters
    ----------
    specification : dictionary or `pyrameter.Scope`
        The specification of the hyperparameter search space.

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
    root = '/'
