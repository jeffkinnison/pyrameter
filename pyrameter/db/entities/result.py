"""A hyperparameter search result.

Classes
-------
Result

"""

from tesserae.db.entities.entity import Entity


class Result(Entity):
    """Data model for a hyperparameter search result.

    Attributes
    ----------
    loss : float
        The value being optimized.
    values
        The hyperparameter values that generated this result.
    additional_metrics : dict
        Dictionary of additional preformance metrics to record alongside this
        result.

    """

    def __init__(self, loss=None, values=None, additional_metrics=None):
        super(Result, self).__init__()
        self.loss = loss
        try:
            iter(values)
            self.values = values
        except TypeError:
            self.values = [values] if values is not None else []

        self.additional_metrics =
            additional_metrics if additional_metrics is not None else {}

    def to_json(self):
        """Serialize this entity as JSON.

        Returns
        -------
        serialized : dict
            A JSON-serializable dictionary representation of this entity.
        """
        return {
            'loss': self.loss,
            'values': self.values,
            'additional_metrics': self.additional_metrics
        }
