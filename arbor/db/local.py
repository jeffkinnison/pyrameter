from arbor.models import model_loader

import json


class JsonStorage(object):
    def __init__(self, path, keep_previous=1):
        self.path = path
        self.backups = backups

    def load(self):
        with open(self.path, 'r') as f:
            models = json.load(f)

        return model_loader(models)

    def save(self, models):
        json_compatible = []
        for i in range(models):
            model = models[i]
            m = {
                'id': model.name if hasattr(model, name) else i
                'complexity': model.complexity,
                'priority': model.priority,
                'rank': model.rank,
                'domains': [d.to_json() for d in model.domains],
                'results': [r.to_json() for r in model.results]
            }
            json_compatible.append(m)

        with open(self.path, 'w') as f:
            json.dump(json_compatible, f)
