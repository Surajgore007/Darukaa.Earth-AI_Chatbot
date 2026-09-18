"""
Database connection and client utilities for Supabase PostgreSQL with pgvector.
"""

import os
from contextlib import contextmanager
from typing import Generator, Optional
import psycopg
from psycopg.rows import dict_row
from pgvector.psycopg import register_vector

from app.core.config import settings


@contextmanager
def get_db_connection(register_pgvector: bool = True) -> Generator[psycopg.Connection, None, None]:
    """
    Context manager that yields an active psycopg3 connection to Supabase.
    Gracefully registers pgvector extension types if the extension exists.
    """
    if not settings.is_database_configured:
        raise ConnectionError(
            "DATABASE_URL is not configured. Please provide your Supabase connection string in .env."
        )

    conn = psycopg.connect(
        conninfo=settings.database_url,
        row_factory=dict_row,
        autocommit=False
    )
    try:
        if register_pgvector:
            try:
                # Register pgvector type adapter with the connection
                register_vector(conn)
            except psycopg.ProgrammingError:
                # The vector extension might not be enabled yet on a fresh database
                pass
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def test_db_connection() -> bool:
    """
    Tests if the database connection can be established.
    Returns True on success, False on failure.
    """
    if not settings.is_database_configured:
        return False
    try:
        with get_db_connection(register_pgvector=False) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")
                row = cur.fetchone()
                return bool(row and row["?column?"] == 1 or row and list(row.values())[0] == 1)
    except Exception:
        return False


def initialize_schema(schema_file_path: Optional[str] = None) -> bool:
    """
    Executes schema.sql against the database to create tables and indexes.
    Enables vector and uuid extensions first before applying table schemas.
    """
    if not schema_file_path:
        schema_file_path = os.path.join(os.path.dirname(__file__), "schema.sql")

    with open(schema_file_path, "r", encoding="utf-8") as f:
        sql_commands = f.read()

    # Step 1: Connect without vector type adapter to enable extension
    with get_db_connection(register_pgvector=False) as conn:
        with conn.cursor() as cur:
            cur.execute('CREATE EXTENSION IF NOT EXISTS vector;')
            cur.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp";')
        conn.commit()

    # Step 2: Now that the vector extension is created, register vector and create tables
    with get_db_connection(register_pgvector=True) as conn:
        with conn.cursor() as cur:
            cur.execute(sql_commands)
        conn.commit()
    return True
