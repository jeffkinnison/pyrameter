import warnings


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn


def boxplot(trials, value='objective', labels='id', filename=None, **kwargs):
    """Create boxplot(s) of hyperparameter performance.

    Boxplots show the distribution of a variable recorded in each trial
    (e.g., loss, accuracy, etc.) with respect to some common feature
    (e.g., algorithm/architecture used, hyperparameter value, etc.). This
    function will create a boxplot with boxes split along user-specified
    related to either the hyperparameter trees being searched or the
    values of hyperparameters searched.

    Parameters
    ----------
    trials : list of `pyrameter.trial.Trial`
        The trials to plot.
    value : str
        The varying value to plot. Pass 'objective' to plot the distribution
        of the objective function; otherwise the ``Trial.results`` dictionary
        will be searched for a key matching the supplied value. To indicate a
        nested value in ``Trial.results``, separate keys with a '.', for
        example 'foo.bar' to use ``results['foo']['bar']``.
        Default: 'objective'.
    labels : str
        The values to bin ``value`` by. Pass 'id' to plot by hyperparameter
        search space; otherwise, the ``Trial.hyperparameters`` dictionary
        will be searched for a key matching the supplied value. To indicate
        a nested value in ``Trial.hyperparameters``, separate keys with a '.',
        for example 'foo.bar' to use ``hyperparameters['foo']['bar']``.
        Default: 'id'.
    filename : str, optional
        If a filename is given, the plot will be saved to that location.
    
    Other Parameters
    ----------------
    **kwargs
        Additional arguments to be passed to `seaborn.boxplot`.

    See Also
    --------
    [`seaborn.boxplot`](https://seaborn.pydata.org/generated/seaborn.boxplot.html#seaborn.boxplot)

    Examples
    --------

    """
    if labels != 'id':
        pass
    else:
        distinct = list(set([t.searchspace().id
                             if t.searchspace is not None else None
                             for t in trials]))
                                




def scatterplot(trials, x):
    obj = [t.objective for t in trials if t.objective is not None]
    x = [t.parameter_dict[x] for t in trials if t.objective is not None]

    sns.scatterplot(x=x, y=obj)
    plt.show()

def heatmap(trials, x, y):
    obj = [t.objective for t in trials if t.objective is not None]
    x = [t.parameter_dict[x] for t in trials if t.objective is not None]
    y = [t.parameter_dict[y] for t in trials if t.objective is not None]

    n_bins = int(math.ceil((2 * math.pi) / 0.1))
    hist, ybins, xbins = np.histogram2d(y, x, bins=n_bins)
    weights, ybins, xbins = np.histogram2d(y, x, bins=n_bins, weights=obj)
    weights /= hist

    sns.heatmap(weights, vmin=-2, vmax=2, xticklabels=xbins, yticklabels=ybins)
    plt.show()
