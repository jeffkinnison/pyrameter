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
- `dill`
- `six`

## A Short Example

```python
import math
import pyrameter

# Minimize the sin function
def objective(params):
    return math.sin(params['x'])

# Uniformly sample values over [0, pi]
space = {
    'x': pyrameter.uniform(0, math.pi),

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
```
