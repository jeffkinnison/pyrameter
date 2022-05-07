"""Scaffolding to standardize optimization method development.

Classes
-------
Method
    Abstract class on which to develop optimization methods.
PopulationMethod
    Abstract class on which to develop population-based optimization methods.
BilevelMethod
    Abstract class on which to develop bilevel optimization methods.
PopulationBilevelMethod
    Abstract class on which to develop bilevel population-based optimization methods.
"""
import copy
import inspect
from queue import LifoQueue
import uuid

import numpy as np


class Method():
    """Abstract class on which to develop optimization methods.

    To create a new method, implement ``__init__`` with custom intializaiton
    args and ``generate`` with the signature defined here.

    Parameters
    ----------
    warm_up : int
        The number of warm up trials to run to prime the optimization method.
        Default: ``20``
    
    Attributes
    ----------
    id : str
        Internal ID used for storage purposes.
    warm_up : int
        The number of warm up trials to run to prime the optimization method.
    """
    def __init__(self, warm_up=20):
        self.id = str(uuid.uuid4())
        self.warm_up = warm_up
        self.n_generated = 0
        self.parameter_queue = LifoQueue()

    def __call__(self, space):
        """Handler for generating hyperparameters.

        Parameters
        ----------
        space : pyrameter.searchspace.SearchSpace
            The search space from which to draw (hyper)parameters.
        
        Returns
        -------
        parameters : array-like
            The generated (hyper)parameters in the same order as
            ``space.domains``, normalized to their respective domains.

        Notes
        -----
        Do not override this method, as it does bookkeeping for pyrameter.
        Instead, override `Method.generate` in subclasses to implement the
        optimization method.
        """
        if self.parameter_queue.empty():
            # Put the hyperparameters and objective values into an array
            trial_data = space.to_array()
            completed = trial_data.shape[0] if trial_data is not None else 0

            # Either randomly generate seed hyperparameters for guided methods
            # or generate values from this method. ``low_check`` ensures we
            # randomly sample at least ``warm_up`` hyperparameter sets, and
            # ``mod_check`` injects random samples to mitigate local minima.
            low_check = self.n_generated < self.warm_up
            mod_check = (self.n_generated % self.warm_up) == 0
            
            if completed < 2 or low_check or mod_check:
                parameters = space.generate()
            else:
                parameters = self.generate(trial_data, space.domains)

            if parameters.ndim == 1:
                parameters = np.expand_dims(parameters, axis=0)

            self.n_generated += parameters.shape[0]

            # Make sure the returned parameters are 1) in some type of sequence
            # and 2) the same length as the number of domains. Mismatches here
            # break future generation.
            # try:
            #     if len(parameters) != len(space.domains):
            #         raise ValueError(
            #             'Generated parameter sequence of length ' +
            #             f'{len(parameters)} does not match the number of ' +
            #             f'domains ({len(space.domains)}). Please ensure the ' +
            #             'method generates one hyperparameter per domain.'
            #         )
            # except TypeError:
            #     raise ValueError(
            #         'Generated parameters must be a sequence or array of ' +
            #         f'values, received {type(parameters)} from method ' +
            #         f'{self.__name__}. Please ensure the method output ' +
            #         'is correct.'
            #     )
            
            for i in range(parameters.shape[0]):
                self.parameter_queue.put_nowait(parameters[i])

        if not self.parameter_queue.empty():
            parameters = self.normalize(space, self.parameter_queue.get_nowait())

        return parameters

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
            List of normalized hyperparameters in the same order as
            ``hyperparameters``.
        """
        return [d.bound_index(h)
                for (d, h) in zip(space.domains, hyperparameters)]

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
        trials to run per generation. Default: 50.
    """
    def __init__(self, population_size=50):
        super().__init__(warm_up=population_size)
        self.population_size = population_size

    def __call__(self, space):
        """Handler for generating populations of hyperparameters.

        Parameters
        ----------
        space : pyrameter.searchspace.PopulationSearchSpace
            The search space from which to draw (hyper)parameters.
        
        Returns
        -------
        parameters : array-like
            The generated (hyper)parameters in the same order as
            ``space.domains``, normalized to their respective domains.
        
        Notes
        -----
        Do not override this method, as it does bookkeeping for pyrameter.
        Instead, override `PopulationMethod.generate` in subclasses to
        implement the optimization method.

        If no population has been generated, this method will default to
        randomly generating an initial population from ``space``.
        """
        # Get the current population. If the search space does not support
        # populations, report this and shut down.
        try:
            current = space.population
        except AttributeError:
            raise TypeError(
                f'Provided search space {space.__name__} is not a ' +
                'PopulationSearchSpace. Are you sure you are working with ' +
                'a population-based optimizer?'
            )

        # Randomly generate an initial population or update the current population.
        if space.population is None:
            new_population = [space.generate() for _ in range(self.population_size)]
        else:
            new_population = self.generate(space.population_to_array(), space.domains)

        self.n_generated += len(new_population)

        try:
            if len(new_population) != self.population_size:
                raise ValueError(
                    f'Generated population of size {len(new_population)} ' +
                    'does not match the specified population size of ' +
                    f'{self.population_size}. Please ensure that this is ' +
                    'correct behavior and update ``self.population_size`` ' +
                    'accordingly.'
                )

            lens = [len(p) == len(space.domains) for p in new_population]

            if not all(lens):
                raise ValueError(
                    'Generated parameter sequence length does not match ' +
                    f'the number of domains ({len(space.domains)}) for one ' +
                    'or more members of the new population. Please ensure ' +
                    f'the method {self.__class__} generates one parameter ' +
                    'per domain for each member of the population.'
                )
        except TypeError:
            raise ValueError(
                'Generated population of parameters must be a 2D sequence ' +
                f'or array of values, received {type(new_population)} from ' +
                f'method {self.__class__}. Please ensure the method output ' +
                'is correct.'
            )

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
    args and ``generate`` with the signature defined in
    ``pyrameter.methods.method.Method``.

    Parameters
    ----------
    inner_method : `pyrameter.methods.method.Method`
        The inner method to call, either the subclass of ``Method`` itself or
        an instance of a subclass of ``Method``. If not instantiated, will be
        instantiated on BilevelMethod construction.
    warm_up : int
        The number of warm up trials to run to prime the optimization method.
        Default: ``20``.

    Attributes
    ----------
    inner_method : `pyrameter.methods.method.Method`
        The inner method to call.
    warm_up : int
        The number of warm up trials to run to prime the optimization method.

    Notes
    -----
    ``inner_method`` should be used in the override of
    `BilevelMethod.generate` to perform the inner optimization loop.
    """
    def __init__(self, inner_method=None, warm_up=20):
        super().__init__(warm_up=warm_up)
        
        # Instantiate a Method subclass with default args if the class itself
        # is passed.
        if inspect.isclass(inner_method) and issubclass(inner_method, Method):
            inner_method = inner_method()

        # An inner method is required to use BilevelMethod, so raise an
        # exception if none is provided. This is here because it isn't clear
        # if a fall-back to random search is the better option.
        if inner_method is None:
            raise ValueError(
                'No inner method provided. Pass a function or a subclass of ' +
                'pyrameter.methods.method.Method when creating BilevelMethod.')
        
        # If the provided inner method is not a subclass of Method or is not
        # callable, inform the user that optimization methods need to actually
        # generate values.
        if not isinstance(inner_method, Method) or not callable(inner_method):
            raise ValueError(
                f'Provided inner method {inner_method} is not a subclass ' +
                 'of pyrameter.methods.Method nor a valid callable.')
        
        self.inner_method = inner_method


class BilevelPopulationMethod(PopulationMethod):
    """Abstract class on which to develop bilevel population-based optimization methods.

    To create a new method, implement ``__init__`` with custom initialization
    args and ``generate`` with the signature defined in
    ``pyrameter.methods.method.PopulationMethod``.

    Parameters
    ----------
    inner_method : `pyrameter.methods.method.Method`
        The inner method to call, either the subclass of ``Method`` itself or
        an instance of a subclass of ``Method``. If not instantiated, will be
        instantiated on BilevelMethod construction.
    population_size : int
        The size of the population to maintain and update, e.g. the number of
        trials to run per generation. Default: 50.
    
    Attributes
    ----------
    inner_method : `pyrameter.methods.method.Method`
        The inner method to call.
    population_size : int
        The size of the population to maintain and update.

    Notes
    -----
    ``inner_method`` should be used in the override of
    `BilevelMethod.generate` to perform the inner optimization loop.
    """
    def __init__(self, inner_method=None, population_size=50):
        super().__init__(population_size=population_size)

        # Instantiate a PopulationMethod subclass with default args if the
        # class itself is passed.
        if inspect.isclass(inner_method) and issubclass(inner_method, PopulationMethod):
            inner_method = inner_method()

        # An inner method is required to use BilevelMethod, so raise an
        # exception if none is provided. This is here because it isn't clear
        # if a fall-back to random search is the better option.
        if inner_method is None:
            raise ValueError(
                'No inner method provided. Pass a function or a subclass of ' +
                'pyrameter.methods.method.Method when creating BilevelMethod.')
        
        # If the provided inner method is not a subclass of Method or is not
        # callable, inform the user that optimization methods need to actually
        # generate values.
        if not isinstance(inner_method, PopulationMethod) or not callable(inner_method):
            raise ValueError(
                f'Provided inner method {inner_method} is not a subclass ' +
                 'of pyrameter.methods.Method nor a valid callable.')
        
        self.inner_method = inner_method