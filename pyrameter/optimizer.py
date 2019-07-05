"""Tools to optimize hyperparameters drawn from a search space.

Classes
-------
FMin
    Minimize an objective function to optimize a set of hyperparameters.
"""


class FMin(object):
    """Minimize an objective function to optimize a set of hyperparameters.

    Parameters
    ----------
    spec
        Specification of the hyperparameter domains to optimize.
    method : {'random','tpe','spearmint','smac'}
        Hyperparameter selection method.
    backend
        Data storage backend.

    Attributes
    ----------
    """

    def __init__(self, spec, method, backend):
        pass

    def generate(self):
        pass

    def optimum(self):
        pass

    def sort_spaces(self):
        pass
