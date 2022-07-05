Parameter Domains
=================

To make searches as accessible as possible, `pyrameter` provides a number of
different types of domains. These can be composed to make trees of domains and
dependencies, allowing complex searches to be expressed as small data structures.


Constant Domains
----------------

The simplest domain has only a single value, and remains constant throughout
the search. Any Python builtin number or string can be passed to the optimizer,
and it will be considered a constant domain in the search::

    import pyrameter

    # Define constants as Python numbers or strings, or use the pyrameter object
    space = {
        'x': 2.7,
        'y': 'foo',
        'z': pyrameter.ConstantDomain(-347)
    }

    opt = pyrameter.FMin('constants', space)

    opt.generate()

    # This will always generate the dictionary
    # {'x': 2.7, 'y': 'foo', 'z': -347}


Discrete Domains
----------------

Discrete domains represent discrete or categorical spaces, which can be used
to search over parameters like neural network activation functions or optimizers.

::

    import pyrameter

    # Define discrete domains as Python lists, or use the pyrameter object.
    space = {
        'x': [1, 2, 3],
        'y': ['relu', 'softmax', 'tanh', 'sigmoid'],
        'z': pyrameter.DiscreteDomain([1, 'foo', -0.576, False, None])
    }

    opt = pyrameter.FMin('constants', space)

    opt.generate()

Every call to `opt.generate()` will generate one value from each of the three
named domains `x`, `y`, and `z`. In discrete domains, data types may be mixed,
and can even be unrelated.


Continuous Domains
------------------

Continuous domains are continuous probability distributions like the Gaussian,
gamma, or Weibull distributions. Defining continuous domains creates a search
with an infinite number of possible parameter values.

::

    import scipy.stats
    import pyrameter

    # Define using the pyrameter helper functions, the scipy.stats interface
    # or the pyrameter object
    space = {
        'x': pyrameter.uniform(3, 845),
        'y': pyrameter.normal(0, 1),
        'z': scipy.stats.vonmises(3.99),
        'aa': pyrameter.ContinuousDomain('gamma', 1.99)
        'bb': pyrameter.ContinuousDomain(scipy.stats.weibull_min, 1.79)
    }

    opt = pyrameter.FMin('constants', space)

    opt.generate()

Every call to `opt.generate()` will generate one floating point number for each
of the four domains. Note that `pyrameter.ContinuousDomain` is compatible with
any `scipy.stats`_ distribution and takes the exact same signature of arguments
and keyword arguments.

.. _scipy.stats: https://docs.scipy.org/doc/scipy/reference/stats.html


Sequences of Domains
--------------------

Domains can be organized into ordered sequences by placing multiple domains in a
tuple or using `pyrameter.domains.SequentialDomain`.

::

    import pyrameter

    # Define constants as Python numbers or strings, or use the pyrameter object
    space = {
        'shape': ([1, 3, 5], [1, 3, 5]),
        ''
    }

    opt = pyrameter.FMin('constants', space)

    opt.generate()




Hierarchical Domains
--------------------


