class InvalidModelError(Exception):
    """Raised when an invalid moded is suppled to get_model_class."""
    def __init__(self, model):
        msg = 'The supplied model {} is not a valid model type.'.format(model)
        msg += '\nValid inputs include subclasses of pyrameter.models.Model,'
        msg += '\n "random", "tpe", and "gp".'
        super(InvalidModelError, self).__init__(msg)


def get_model_class(model):
    """Get model objects based on common names.

    Parameters
    ----------
    model : instance of `pyrameter.models.Model` or {"random", "tpe", "gp"}
        They type of model to retrieve.

    Returns
    -------
    model_class : pyrameter.models.Model
        The requested model class.

    Raises
    ------
    InvalidModelError
        Raised when an invalid moded is suppled to get_model_class.
    """
    from pyrameter.models.model import Model
    from pyrameter.models.random_search import RandomSearchModel
    from pyrameter.models.tpe import TPEModel
    from pyrameter.models.gp import GPBayesModel

    if isinstance(model, Model):
        return model.__class__
    elif isinstance(model, str):
        if model in [u'random', RandomSearchModel.__name__]:
            model = RandomSearchModel
        elif model in [u'tpe', TPEModel.__name__]:
            model = TPEModel
        elif model in [u'gp', GPBayesModel.__name__]:
            model = GPBayesModel
        else:
            raise InvalidModelError(model)
    else:
        raise InvalidModelError(model)

    return model
