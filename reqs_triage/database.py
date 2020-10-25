import os

import psycopg2


def connect():
    """Connects to the database specified by the environmental variables."""
    host = os.environ.get("PG_HOST", "localhost")
    port = os.environ.get("PG_PORT", "5432")
    db = os.environ.get("PG_DB", "postgres")
    user = os.environ.get("PG_USER", "postgres")
    return psycopg2.connect(f"dbname={db} user={user} host={host} port={port}")


def execute_sql(sql, connection, values=None, select=False, commit=True):
    """Executes a SQL statement against the database.

    Parameters
    ----------
    sql : str
        The SQL statement to execute
    connection : psycopg2.connection
        The database connection
    select : bool
        If True, returns the query results as tuples
    commit : bool
        Determines whether or not to commit the database operation upon completion
    """
    if not connection:
        connection = connect()
    with connection.cursor() as cursor:
        cursor.execute(sql, values)
        if select:
            return cursor.fetchall()
    if commit:
        connection.commit()
