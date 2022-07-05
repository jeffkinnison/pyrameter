"""SQL backend storage engine.

Classes
-------
SQLiteBackend
    Save search results to a SQLite database.
"""

import sqlite3

from pyrameter.backend.base import BaseBackend


class SQLiteBackend(BaseBackend):
    """Save search results to a SQLite database.

    Parameters
    ----------
    database : str
        Path to the dtabase file.
    """

    def __init__(self, database):
        self.database = database
        connection = sqlite3.connect(database)

        connection.execute('''
            create table if not exists experiments(
                id integer primary key,
                exp_key varchar)''')
        connection.execute('''
            create table if not exists searchspaces(
                id integer primary key,
                foreign key(experiment) references experiments(id),
                complexity real,
                uncertainty real)''')
        connection.execute('''
            create table if not exists domains(
                id integer primary key,
                foreign key(searchspace) references searchspaces(id),
                domain text,
                rng text)''')
        connection.execute('''
            create table if not exists trials(
                id integer primary key,
                foreign key(searchspace) references searchspaces(id),
                status integer,
                objective real,
                results text,
                errmsg text)''')

    def load(self, exp_key):
        connection = sqlite3.connect(database)

        experiment = connection.execute(
            'SELECT * from experiments WHERE exp_key=:key',
            {'key': exp_key})

        searchspaces = connection.execute(
            'SELECT * from searchspaces WHERE experiment=:expid}',
            {'expid': experiment.id}
        )

        for searchspace in searchspaces:
            


    def save(self, searchspaces):
        pass
