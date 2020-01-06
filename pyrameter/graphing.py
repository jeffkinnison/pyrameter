import seaborn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
