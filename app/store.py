"""Storage: Postgres on Railway (DATABASE_URL = postgresql://...), SQLite for local runs and tests
(DATABASE_URL = sqlite:///path, or nothing: a file in the working folder). The same SQL works on both.

    installs  one row per plugin that shares (a random id it made itself; nothing else about it)
    answers   its latest answer per sound: the sound's file hash, the kind, its pack (a hash), the scheme of the
              features, the measurements (fp) and the rows the app made of it (each check's label + features, JSON)
    models    every model trained: candidate / rejected / published / revoked, its text, signature, report, gate
    state     small things to remember (when it last trained)
"""
from __future__ import annotations

import json
import sqlite3
import threading
from typing import Any, Callable, Iterable

SCHEMA = [
    "CREATE TABLE IF NOT EXISTS installs (id TEXT PRIMARY KEY, first_seen BIGINT NOT NULL, last_seen BIGINT NOT NULL, app TEXT NOT NULL)",
    "CREATE TABLE IF NOT EXISTS answers (install TEXT NOT NULL, hash TEXT NOT NULL, kind TEXT NOT NULL, pack TEXT NOT NULL, "
    "scheme INTEGER NOT NULL, fp TEXT NOT NULL, rows TEXT NOT NULL, app TEXT NOT NULL, received BIGINT NOT NULL, PRIMARY KEY (install, hash))",
    "CREATE INDEX IF NOT EXISTS answers_hash ON answers (hash)",
    "CREATE TABLE IF NOT EXISTS models (version INTEGER PRIMARY KEY, scheme INTEGER NOT NULL, created BIGINT NOT NULL, status TEXT NOT NULL, "
    "text TEXT NOT NULL, signature TEXT NOT NULL, report TEXT NOT NULL, gate TEXT NOT NULL, answers INTEGER NOT NULL)",
    "CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
]


class Store:
    def __init__(self, url: str | None):
        self.url = url or "sqlite:///samplemanager-cloud.db"
        self.pg = self.url.startswith("postgres")
        self.lock = threading.Lock()
        self.conn: Any = None
        self._connect()
        for sql in SCHEMA:
            self._run(sql)

    # ------------------------------------------------------------ plumbing

    def _connect(self) -> None:
        if self.pg:
            import psycopg

            self.conn = psycopg.connect(self.url, autocommit=True)
        else:
            path = self.url[len("sqlite:///"):] if self.url.startswith("sqlite:///") else ":memory:"
            self.conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)

    def _sql(self, sql: str) -> str:
        return sql.replace("?", "%s") if self.pg else sql

    def _with_conn(self, work: Callable[[Any], Any]) -> Any:
        """work(connection), once more on a new connection if Postgres dropped it (Railway restarts the database now and
        then; a transaction cut off that way was rolled back, so doing it again is safe)."""
        with self.lock:
            for attempt in (1, 2):
                try:
                    return work(self.conn)
                except Exception as e:
                    if not self.pg or attempt == 2 or not (_broken(e) or getattr(self.conn, "broken", False) or getattr(self.conn, "closed", False)):
                        raise
                    try:
                        self.conn.close()
                    except Exception:
                        pass
                    self._connect()

    def _run(self, sql: str, args: Iterable[Any] = (), fetch: bool = False) -> list[tuple]:
        """One statement."""
        def work(conn: Any) -> list[tuple]:
            cur = conn.execute(self._sql(sql), tuple(args))
            return list(cur.fetchall()) if fetch else []
        return self._with_conn(work)

    def _many(self, statements: list[tuple[str, tuple]]) -> None:
        """Several statements as one transaction."""
        def work(conn: Any) -> None:
            if self.pg:
                with conn.transaction():
                    for sql, args in statements:
                        conn.execute(self._sql(sql), args)
            else:
                conn.execute("BEGIN")
                try:
                    for sql, args in statements:
                        conn.execute(sql, args)
                    conn.execute("COMMIT")
                except Exception:
                    conn.execute("ROLLBACK")
                    raise
        self._with_conn(work)

    # ------------------------------------------------------------ answers

    def put_answers(self, install: str, app: str, scheme: int, answers: list[dict], now: int) -> int:
        """The plugin's answers: each replaces its earlier one for that sound; kind None takes it back."""
        st: list[tuple[str, tuple]] = [(
            "INSERT INTO installs (id, first_seen, last_seen, app) VALUES (?, ?, ?, ?) "
            "ON CONFLICT (id) DO UPDATE SET last_seen = excluded.last_seen, app = excluded.app",
            (install, now, now, app))]
        for a in answers:
            if a["kind"] is None:
                st.append(("DELETE FROM answers WHERE install = ? AND hash = ?", (install, a["hash"])))
            else:
                st.append((
                    "INSERT INTO answers (install, hash, kind, pack, scheme, fp, rows, app, received) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT (install, hash) DO UPDATE SET kind = excluded.kind, pack = excluded.pack, scheme = excluded.scheme, "
                    "fp = excluded.fp, rows = excluded.rows, app = excluded.app, received = excluded.received",
                    (install, a["hash"], a["kind"], a["pack"], scheme, json.dumps(a["fp"], separators=(",", ":")),
                     json.dumps(a["rows"], separators=(",", ":")), app, now)))
        self._many(st)
        return len(answers)

    def delete_install(self, install: str) -> int:
        n = self._run("SELECT COUNT(*) FROM answers WHERE install = ?", (install,), fetch=True)[0][0]
        self._many([("DELETE FROM answers WHERE install = ?", (install,)), ("DELETE FROM installs WHERE id = ?", (install,))])
        return int(n)

    def answers(self, scheme: int, sandbox: bool = True) -> list[tuple[str, str, str, str, list[dict], int]]:
        """(install, hash, kind, pack, rows, received) of every answer with these features, newest first (sandbox=False:
        not the SANDBOX's, its app ends in "sandbox": tests, never taught)."""
        # (the pattern as an argument: in the SQL itself Postgres would read its % as a placeholder)
        where, args = ("scheme = ?", (scheme,)) if sandbox else ("scheme = ? AND app NOT LIKE ?", (scheme, "%sandbox"))
        got = self._run(f"SELECT install, hash, kind, pack, rows, received FROM answers WHERE {where} ORDER BY received DESC",
                        args, fetch=True)
        return [(r[0], r[1], r[2], r[3], json.loads(r[4]), int(r[5])) for r in got]

    def newest_answer(self, scheme: int) -> int:
        got = self._run("SELECT MAX(received) FROM answers WHERE scheme = ?", (scheme,), fetch=True)
        return int(got[0][0] or 0)

    def stats(self) -> dict:
        installs = self._run("SELECT COUNT(*) FROM installs", fetch=True)[0][0]
        answers = self._run("SELECT COUNT(*) FROM answers", fetch=True)[0][0]
        kinds = self._run("SELECT kind, COUNT(*) FROM answers GROUP BY kind ORDER BY COUNT(*) DESC", fetch=True)
        sounds = self._run("SELECT COUNT(DISTINCT hash) FROM answers", fetch=True)[0][0]
        return {"installs": int(installs), "answers": int(answers), "sounds": int(sounds), "kinds": {k: int(n) for k, n in kinds}}

    # ------------------------------------------------------------ models

    def next_version(self) -> int:
        got = self._run("SELECT MAX(version) FROM models", fetch=True)
        return int(got[0][0] or 0) + 1

    def add_model(self, m: dict) -> None:
        self._run("INSERT INTO models (version, scheme, created, status, text, signature, report, gate, answers) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                  (m["version"], m["scheme"], m["created"], m["status"], m["text"], m["signature"], m["report"], m["gate"], m["answers"]))

    def set_status(self, version: int, status: str) -> None:
        self._run("UPDATE models SET status = ? WHERE version = ?", (status, version))

    def model(self, version: int) -> dict | None:
        got = self._run("SELECT version, scheme, created, status, text, signature, report, gate, answers FROM models WHERE version = ?",
                        (version,), fetch=True)
        return _model(got[0]) if got else None

    def published(self, scheme: int) -> dict | None:
        got = self._run("SELECT version, scheme, created, status, text, signature, report, gate, answers FROM models "
                        "WHERE status = 'published' AND scheme = ? ORDER BY version DESC LIMIT 1", (scheme,), fetch=True)
        return _model(got[0]) if got else None

    def models(self) -> list[dict]:
        got = self._run("SELECT version, scheme, created, status, gate, answers FROM models ORDER BY version DESC", fetch=True)
        return [{"version": r[0], "scheme": r[1], "created": r[2], "status": r[3], "gate": json.loads(r[4]), "answers": r[5]} for r in got]

    # ------------------------------------------------------------ state

    def get(self, key: str, default: str = "") -> str:
        got = self._run("SELECT value FROM state WHERE key = ?", (key,), fetch=True)
        return got[0][0] if got else default

    def set(self, key: str, value: str) -> None:
        self._run("INSERT INTO state (key, value) VALUES (?, ?) ON CONFLICT (key) DO UPDATE SET value = excluded.value", (key, value))


def _model(r: tuple) -> dict:
    return {"version": r[0], "scheme": r[1], "created": r[2], "status": r[3], "text": r[4], "signature": r[5],
            "report": r[6], "gate": json.loads(r[7]), "answers": r[8]}


def _broken(e: Exception) -> bool:
    """The connection is gone (psycopg: OperationalError and its kinds, e.g. AdminShutdown; InterfaceError)."""
    try:
        import psycopg
        if isinstance(e, (psycopg.OperationalError, psycopg.InterfaceError)):
            return True
    except ImportError:
        pass
    return "closed" in str(e).lower()
