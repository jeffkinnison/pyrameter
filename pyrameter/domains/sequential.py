"""Hyperparameter domain with multiple domains to generate in order at a time.

Classes
-------
SequentialDomain


"""

from pyrameter.db.entities import Domain


class SequentialDomain(Domain):
    """A sequence of domains to generate values from.

    The SequentialDomain represents a tuple of related domains, for example
    representing the dimensions of a kernel or a sequence of data preprocessing
    operaitons.

    Parameters
    ----------
        domains : tuple of Domain
            The domains to include in this sequence in the order they should be
            generated.
        path : str
            Path to this domain in the search hierarchy.
        callback : function or list of function
            A callback to apply to each generated value.

    """

    def __init__(self, domains, path='', callback=None):
        self.callback = callback if callback is not None else (lambda x: x)

        if not isinstance(self.callback, list) and self.callback is not None:
            self.callback = [self.callback for _ in range(len(domains))]
        elif len(self.callback) < len(domains):
            msg = "Number of callbacks ({}) does not" + \
                  " match number of domains ({})"
            raise ValueError(msg.format(len(callback), len(domains)))

        super(SequentialDomain, self).__init__(domain=domains, path=path)

    @property
    def complexity(self):
        return sum([d.complexity for d in self.domain])

    def generate(self):
        return (self.callback(d.generate()) for d in self.domain)

    def to_json(self):
        return {
            'domain': [d.to_json() for d in self.domain],
            'path': self.path
        }
