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
        self.connection = sqlite3.connect(database)

        self.connection.execute('''
            create table if not exists searchspaces(
                id integer primary key,
                exp_key varchar,
                complexity real,
                uncertainty real)''')
        self.connection.execute('''
            create table if not exists domains(
                id integer primary key,
                foreign key(searchspace) references searchspaces(id),
                domain text,
                rng text)''')
        self.connection.execute('''
            create table if not exists trials(
                id integer primary key,
                foreign key(searchspace) references searchspaces(id),
                status integer,
                objective real,
                results text,
                errmsg text)''')

    def load(self):
        pass

    def save(self, searchspaces):
        pass
