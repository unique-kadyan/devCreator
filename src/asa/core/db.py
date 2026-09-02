"""One place that knows how to open the database correctly.

Every module used to carry its own `sqlite3.connect`, which meant foreign keys were
enforced in some paths and silently ignored in others. WAL + busy_timeout matter because
the dashboard reads while the runner writes; without them a page refresh during a render
raises "database is locked".
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

MIGRATIONS = Path(__file__).resolve().parents[3] / "migrations"


def connect(db: Path | str) -> sqlite3.Connection:
    con = sqlite3.connect(str(db), timeout=30.0, isolation_level=None)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA foreign_keys=ON")
    con.execute("PRAGMA busy_timeout=30000")
    con.execute("PRAGMA synchronous=NORMAL")
    return con


@contextmanager
def tx(db: Path | str) -> Iterator[sqlite3.Connection]:
    """Explicit transaction. Rolls back on any exception, then re-raises."""
    con = connect(db)
    try:
        con.execute("BEGIN IMMEDIATE")
        yield con
        con.execute("COMMIT")
    except BaseException:
        try:
            con.execute("ROLLBACK")
        except sqlite3.Error:
            pass
        raise
    finally:
        con.close()


@contextmanager
def read(db: Path | str) -> Iterator[sqlite3.Connection]:
    con = connect(db)
    try:
        yield con
    finally:
        con.close()


def migrate(db: Path | str, migrations_dir: Path = MIGRATIONS) -> list[str]:
    """Apply every *.sql in name order exactly once. Returns the ones applied now.

    Migrations must be idempotent at statement level (CREATE TABLE IF NOT EXISTS etc.)
    because an interrupted run may have applied part of a file before the ledger row
    was written.
    """
    Path(db).parent.mkdir(parents=True, exist_ok=True)
    con = connect(db)
    try:
        con.execute("CREATE TABLE IF NOT EXISTS schema_migrations ("
                    "name TEXT PRIMARY KEY, applied_at TEXT NOT NULL "
                    "DEFAULT (datetime('now')))")
        done = {r["name"] for r in con.execute("SELECT name FROM schema_migrations")}
        applied: list[str] = []
        for path in sorted(Path(migrations_dir).glob("*.sql")):
            if path.name in done:
                continue
            con.executescript(path.read_text())
            con.execute("INSERT OR IGNORE INTO schema_migrations(name) VALUES (?)",
                        (path.name,))
            applied.append(path.name)
        return applied
    finally:
        con.close()


def jload(value: Any, default: Any = None) -> Any:
    """Decode a JSON column that may be NULL, empty or already-decoded."""
    if value is None or value == "":
        return default if default is not None else {}
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default if default is not None else {}


def jdump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
