.. pyrameter documentation master file, created by
   sphinx-quickstart on Fri Oct 12 16:58:14 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

pyrameter
=========

Optimization tasks, especially in machine learning, can incorporate a wide
variety of domains to search. Hyperparameter searches, in particular,
can cover a wide variety of continuous and categorical, dependent and disjoint
search spaces that must be jointly optimized.

`pyrameter` is a package for defining, running, and evaluating (hyper)parameter
optimization tasks. Within, standard types of domains, search strategies, and
storage backends are unified to make setting up and running these large
parameter sweeps easy and flexible.

Installation
------------

`pip`

::

    pip install pyrameter

A Basic Example
---------------

::

    import math
    import pyrameter

    # Minimize the sin function
    def objective(params):
        return math.sin(params['x'])

    # Uniformly sample values over [0, pi]
    space = {
        'x': pyrameter.ContinuousDomain('uniform', loc=0, scale=math.pi)
    }

    # Set up the search with an experiment key, the domains to search, and
    # random search to generate values.
    opt = pyrameter.FMin('sin_exp', space, 'random')

    # Try 1000 values of x and store the result.
    for i in range(1000):
        trial = opt.generate()
        trial.objective = objective(trial.hyperparameters)

    # Print the x that minimized sin
    print(opt.optimum)


Contributing
------------

To contribute, fork our `GitHub repository`_, add your code, and submit a pull
request.

.. _GitHub repository: https://github.com/jeffkinnison/pyrameter

Issues
------

If you find a bug, `submit an issue`_ to our GitHub repository with the full
stack trace and system description.

.. _submit an issue: https://github.com/jeffkinnison/pyrameter/issues

.. toctree::
    :hidden:
    :maxdepth: 2
    :caption: Contents:

    domains
    specifications
    methods
    storage
    api


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
