# SPDX-FileCopyrightText: 2026 Featune contributors
# SPDX-License-Identifier: MIT

"""Transactional, single-writer study history.

Persists JSON records transactionally and provides cross-process writer exclusion.
Raw observations, credentials and fitted models do not belong in SQLite records.

Created:
    2026-09-21
"""

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path


def dumps(value):
    """Serialize persisted records deterministically while rejecting non-finite values.

    Args:
        value (JSON-serializable object): Metadata or trial payload.

    Returns:
        str: Sorted-key Unicode JSON.

    Raises:
        ValueError: Payload contains NaN or infinity.
        TypeError: Payload contains an unsupported object.
    """
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True)


class Storage:
    """Persist study metadata and trials with SQLite transactions and a writer lock.

    Attributes:
        directory (Path): Per-study directory containing database, lock and artifacts.
        path (Path): SQLite database file.

    Notes:
        The advisory OS lock coordinates optimize writers; callers performing direct
        mutations must acquire lock themselves. Connections commit on success and roll
        back on exceptions. SQLite errors are deliberately propagated.
    """

    def __init__(self, directory):
        """Create the study directory and initialize WAL-mode database tables.

        Args:
            directory (str or Path): Local directory owned by this study.
        """
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "study.sqlite3"
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS trials (number INTEGER PRIMARY KEY, value TEXT NOT NULL)")

    @contextmanager
    def connect(self):
        """Open a transaction-scoped connection and always close it.

        Yields:
            sqlite3.Connection: Connection committing on normal exit and rolling back on errors.

        Notes:
            Connection acquisition waits up to 30 seconds for SQLite locks. Errors propagate.
        """
        db = sqlite3.connect(self.path, timeout=30)
        try:
            with db:
                yield db
        finally:
            db.close()

    @contextmanager
    def lock(self):
        # The OS releases this advisory lock even after an unclean process termination.
        """Acquire the process-level nonblocking writer lock for this study.

        Yields:
            None: The caller may mutate study state while inside the context.

        Raises:
            RuntimeError: Another process holds the study writer lock.

        Notes:
            The lock is released in finally and by the OS after process termination.
            Windows locks one byte; POSIX uses flock.
        """
        import os

        with (self.directory / ".writer.lock").open("a+b") as handle:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                handle.write(b"0")
                handle.flush()
                handle.seek(0)
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                except OSError:
                    raise RuntimeError("Another writer is optimizing this study") from None
            else:
                import fcntl

                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    raise RuntimeError("Another writer is optimizing this study") from None
            try:
                yield
            finally:
                if os.name == "nt":
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(handle, fcntl.LOCK_UN)

    def load(self):
        """Read persisted metadata and trials in trial-number order.

        Returns:
            tuple[dict, list[dict]]: Decoded metadata and chronological trial payloads.
        """
        with self.connect() as db:
            metadata = {
                key: json.loads(value) for key, value in db.execute("SELECT key, value FROM metadata")
            }
            trials = [
                json.loads(value) for (value,) in db.execute("SELECT value FROM trials ORDER BY number")
            ]
        return metadata, trials

    def save(self, trial=None, **metadata):
        """Commit one trial and metadata updates in a single transaction.

        Args:
            trial (dict or None): Trial payload containing number; None updates metadata only.
            metadata (Any): JSON-serializable keyword metadata entries.

        Returns:
            None: Existing keys and trial numbers are replaced.
        """
        # Trial state and budget metadata must commit together to keep resume accounting consistent.
        with self.connect() as db:
            if trial is not None:
                db.execute("INSERT OR REPLACE INTO trials VALUES (?, ?)", (trial["number"], dumps(trial)))
            db.executemany(
                "INSERT OR REPLACE INTO metadata VALUES (?, ?)",
                [(key, dumps(value)) for key, value in metadata.items()],
            )

    def archive_and_reset(self, metadata, trials):
        """Archive an incompatible experiment and clear its active records atomically.

        Args:
            metadata (dict): Full previous study metadata.
            trials (list[dict]): Full previous trial payloads.

        Returns:
            None: Adds an archive row and deletes active trials/metadata.

        Notes:
            Does not delete external cache or exported model files. Call while holding lock().
        """
        with self.connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS archives (id INTEGER PRIMARY KEY, value TEXT NOT NULL)")
            db.execute(
                "INSERT INTO archives(value) VALUES (?)", (dumps({"metadata": metadata, "trials": trials}),)
            )
            db.execute("DELETE FROM trials")
            db.execute("DELETE FROM metadata")
