from pyrameter.domain import Domain, DiscreteDomain


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
    exclusive : bool
        If True, split this Scope on each contained Scope or Domain. Default:
        False.
    optional : bool
        If True, split this scope by creating an empty clone. Default: False.
    *args, **kws
        Structures defining the Scope and Domain members of this Scope.

    Attributes
    ----------
    children : list of {Scope,pyrameter.Domain}
        The members of this scope.
    exclusive : bool
        If True, split this Scope on each contained Scope or Domain.
    optional : bool
        If True, split this scope by creating an empty clone.
    """
    def __init__(self, exclusive=False, optional=False, *args, **kws):
        self.exclusive = exclusive
        self.optional = optional

        self.children = {}
        for arg in args:
            self.children[str(arg[0])] = arg[1]

        for kw in kws:
            self.children[str(kw)] = kws[kw]

    def split(self, path=''):
        """Split this scope into its constituent models.


        """
        models = []

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
                m = self.__create_model()
                m.add_domain(cpath, cval)

                # Store as individual models if exclusive, otherwise merge
                if self.exclusive or len(models) == 0:
                    models.append(m)
                else:
                    for model in models:
                        model.merge(m)

        # Create an empty model to account for optional scopes
        if self.optional:
            models.append(Model())

        return models

    def copy(self, with_children=True):
        return Scope(exclusive=self.exclusive,
                     optional=self.optional,
                     **self.children)

    def __create_model(self):
        pass
