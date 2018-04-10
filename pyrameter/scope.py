from pyrameter.domain import Domain, DiscreteDomain
from pyrameter.models import get_model_class


class DuplicateDomainError(Exception):
    def __init__(self, key, orig, dup):
        msg = 'Multiple domains with the same name were passed to a same scope.'
        msg += '\nThe offending domains are: {}:{} and {}:{}'.format(
            key, orig, key, dup)
        msg += '\nRename one of these domains to resolve this issue.'
        super(DuplicateDomainError, self).__init__(msg)


class Scope(object):
    """A container for related hyperparameter domains.

    Hyperparameter domains are related to one another by the learning model
    they parameterize. The Scope class allows for hierarchical hyperparameter
    domain structures to be created by nesting scopes. Scopes may be exclusive
    (only one contained Scope or domain may be used at a time) or optional
    (either use the contained Scopes/domains or exclude them), allowing for
    flexible representations of learning models as tree structures.

    Parameters
    ----------
    children : pair of (str, {Scope,Domain}) or {Scope,Domain}
        The sub-scopes and domains in this scope.
    exclusive : bool
        If True, split this Scope on each contained Scope or Domain. Default:
        False.
    optional : bool
        If True, split this scope by creating an empty clone. Default: False.
    model : {'random','tpe','gp'}
        The search strategy to use.

    Attributes
    ----------
    children : list of {Scope,pyrameter.Domain}
        The members of this scope.
    exclusive : bool
        If True, split this Scope on each contained Scope or Domain.
    optional : bool
        If True, split this scope by creating an empty clone.

    See Also
    --------
    ``pyrameter.Domain``
    ``pyrameter.get_model_class``
    """
    def __init__(self, *args, **kws):
        try:
            self.exclusive = bool(kws.pop('exclusive'))
        except KeyError:
            self.exclusive = False
        
        try:
            self.optional = bool(kws.pop('optional'))
        except KeyError:
            self.optional = False
        
        try:
            model = kws.pop('model')
        except KeyError:
            model = 'random'

        self.children = {}
        for arg in args:
            key, val = arg
            key = str(key)
            if key not in self.children:
                self.children[key] = val
            else:
                raise DuplicateDomainError(key, self.children[key], val)

        for key, val in kws.items():
            if key not in self.children:
                self.children[key] = val
            else:
                raise DuplicateDomainError(key, self.children[key], val)
            
        self.model = model

    def __iter__(self):
        return self.children

    @property
    def model(self):
        return self.__model

    @model.setter
    def model(self, value):
        self.__model = get_model_class(value)
        for key, val in self.children.items():
            if isinstance(val, self.__class__):
                val.model = value

    def split(self, path=''):
        """Split this scope into its constituent models.


        """
        models = [] if self.exclusive else [self.__create_model()]

        for child in self.children:
            # Update the path to the current child in the tree
            cpath = '/'.join([path, child])
            cval = self.children[child]

            # If a Scope, do DFS to process sub-scopes
            if isinstance(cval, Scope):
                submodels = cval.split(path=cpath)

                # If the current scope is exclusive, simple append sub-models.
                # Otherwise merge new sub-models with existing sub-models.
                if self.exclusive:
                    models.extend(submodels)
                else:
                    newmodels = []
                    for model in models:
                        for submodel in submodels:
                            m = model.copy()
                            m.merge(submodel)
                            newmodels.append(m)
                    models = newmodels
            else:
                # Convert non-Domain values into single-value domains
                if not isinstance(cval, Domain):
                    cval = DiscreteDomain(cval)

                # Store the Domain into its own Model to merge later
                cval.path = cpath
                m = self.__create_model()
                m.add_domain(cval)

                # Store as individual models if exclusive, otherwise merge
                if self.exclusive or len(models) == 0:
                    models.append(m)
                else:
                    for model in models:
                        model.merge(m)

        # Create an empty model to account for optional scopes
        if self.optional:
            models.append(self.__create_model())

        return models

    def copy(self, with_children=True):
        return Scope(exclusive=self.exclusive,
                     optional=self.optional,
                     **self.children)

    def __create_model(self):
        return self.model()
