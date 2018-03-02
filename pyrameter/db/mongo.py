from pymongo import MongoClient


class MongoStorage(object):
    def __init__(self, host='localhost', port=27017, db='arbor', exp='arbor',
                 username=None, password=None, **mongo_kws):
        self.client = MongoClient(host, port, username=username,
                                  password=password, **mongo_kws)

        self.db = db
        self.collection = exp

    def save(self, models):
        state = [m.to_json() for model in models]
        
