# pyrameter

`pyrameter` is a library for designing hierarchical parameter searches with
continuous and discrete domains, and then search those spaces.

1. [Installation](#installation)
2. [Dependencies](#dependencies)
3. [A Short Example](#a-short-example)

## Installation

```
$ git clone https://github.com/jeffkinnison/pyrameter
$ cd pyrameter
$ pip install .
```

## Dependencies

- `numpy`
- `scipy`
- `scikit-learn`
- `pymongo`

## A Short Example

```python
import pyrameter.build
from pyrameter import Scope, ContinuousDomain, DiscreteDomain
import scipy.stats

# Define a set of search domains
spec = Scope(x=ContinuousDomain(scipy.stats.uniform, loc=0, scale=1),
             y=DiscreteDomain([i for i in range(100)]))

# Build the model of the search.
model = pyrameter.build(space, method='random', )

# Generate parameter values
params = model.generate()

# params is a dictionary, e.g. {'x': 0.34786, 'y': 27}
```
