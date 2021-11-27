"""Scaffolding to standardize optimization method development.

Classes
-------
Method
    Abstract class on which to develop optimization methods.
PopulationMethod
    Abstract class on which to develop population-based optimization methods.
BilevelMethod
    Abstract class on which to develop bilevel optimization methods.
"""
import copy
import uuid


class Method():
    """Abstract class on which to develop optimization methods.

    To create a new method, implement ``__init__`` with custom intializaiton
    args and ``generate`` with the signature defined here.
    """
    def __init__(self, warm_up=20):
        self.id = str(uuid.uuid4())
        self.warm_up = warm_up

    def __call__(self, space):
        # Put the hyperparameters and objective values into an array
        trial_data = space.to_array()
        completed = trial_data.shape[0] if trial_data is not None else 0

        # Either randomly generate seed hyperparameters for guided methods
        # or generate values from this method.
        if completed <= self.warm_up or completed % self.warm_up == 0:
            parameters = space.generate()
        else:
            parameters = self.generate(trial_data, space.domains)
        
        return self.normalize(space, parameters)

    @classmethod
    def from_json(cls, json_obj):
        """Load method state from a JSON object.

        If any fields in ``json_obj`` need to be reformatted (e.g. if they
        were converted to JSON-compatible format in ``to_json``), override
        this classmethod and do the conversion there, then pass the updated
        dictionary to the default method with
        ``super(<class>, cls).from_json(updated_obj)``, where ``<class>`` is
        the subclass overriding this method.

        Parameters
        ----------
        json_obj : dict
            Method state to load.
        """
        obj = cls()
        obj.__dict__.update(json_obj)
        return obj

    def generate(self, trial_data, domains):
        """Generate a set of hyperparameters.

        Parameters
        ----------
        trial_data : array_like
            A 2-d numpy array where each row is one completed trial
            (hyperparameter set) and each column corresponds to one
            hyperparameter domain (always in the same order) with the
            objective value of the trial in the last column.
        domains : list of pyrameter.domain.base.Domain
            The domains from which hyperparameters were generated. These
            are provided in the same order as the columns in ``trial_data``.
        
        Returns
        -------
        array_like
            A 1-d list or array of new hyperparameter values with one element
            per hyperparameter domain in the same order as the columns in
            ``trial_data``.
        """
        raise NotImplementedError

    def normalize(self, space, hyperparameters):
        """Normalize generated hyperparameters to their respective domains.

        Parameters
        ----------
        space : `pyrameter.searchspace.SearchSpace`
            The search space from which ``hyperparameters`` was generated.
        hyperparameters : array_like
            The set of hyperparameters to normalize, the same length as
            ``space.domains``. Hyperparameter ordering is assumed to match
            the ordering of ``space.domains`` (i.e. ``hyperparameters[i]``
            was drawn from ``space.domains[i]``).
        
        Returns
        -------
        normed : list
            List of normalized hyperparameters in the same order as ``hyperparameters``.
        """
        return [space.domains[i].map_to_domain(h, bound=True)
                for i, h in enumerate(hyperparameters)]

    def to_json(self):
        """Convert method state to a JSON-compatible dictionary.

        The default implementation creates a deep copy of the Method object's
        state dictionary (``self.__dict__``) and returns that. If any
        attributes set in __init__ are not JSON-compatible, override this
        method and convert those attributes to a JSON-compatible format.
        """
        return copy.deepcopy(self.__dict__)


class PopulationMethod(Method):
    """Abstract class on which to develop population-based optimization methods.
    
    To create a new method, implement ``__init__`` with custom initialization
    args and ``generate`` with the signature defined here.

    Parameters
    ----------
    population_size : int
        The size of the population to maintain and update, e.g. the number of
        trials to run per generation.
    """
    def __init__(self, population_size):
        super().__init__()
        self.population_size = population_size

    def __call__(self, space):
        # Get the current population. If the search space does not support
        # populations, report this and shut down.
        try:
            current = space.population
        except AttributeError:
            raise TypeError(f'Provided search space {space} is not a PopulationSearchSpace')

        # Randomly generate an initial population or update the current population.
        if space.population is None:
            new_population = [space.generate() for _ in range(self.population_size)]
        else:
            new_population = self.generate(space.population_to_array(), space.domains)
        
        return [self.normalize(space, p) for p in new_population]

    def generate(self, population_data, domains):
        """Generate a set of hyperparameters.

        Parameters
        ----------
        population_data : array_like
            A 2-d numpy array where each row is one completed trial
            (hyperparameter set) in the current population and each
            column corresponds to one hyperparameter domain (always
            in the same order) with the objective value of the trial
            in the last column.
        domains : list of pyrameter.domain.base.Domain
            The domains from which hyperparameters were generated. These
            are provided in the same order as the columns in ``trial_data``.
        
        Returns
        -------
        array_like
            A 1-d list or array of new hyperparameter values with one element
            per hyperparameter domain in the same order as the columns in
            ``trial_data``.
        """
        raise NotImplementedError


class BilevelMethod(Method):
    """Abstract class on which to develop bilevel optimization methods.

    To create a new method, implement ``__init__`` with custom initialization
    args and ``generate`` with the signature defined here.
    """
    def __init__(self, inner_method):
        super.__init__()
        if not isinstance(inner_method, Method) or not callable(inner_method):
            raise ValueError(f'Provided inner method {inner_method} is not a subclass of pyrameter.methods.Method or valid callable.')
        self.inner_method = inner_method
