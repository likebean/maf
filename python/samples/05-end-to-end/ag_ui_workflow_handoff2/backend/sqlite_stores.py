# Copyright (c) Microsoft. All rights reserved.

"""SQLite stores for the AG-UI handoff2 demo.

Three durable authorities live in one DB file:

- AG-UI Thread Snapshot (UI transcript / hydrate / HITL cards)
- Agent HistoryProvider messages (LLM context), keyed by AgentSession.session_id
- Workflow checkpoints (resume across process restarts)

Resume resolves ``checkpoint_id`` via interrupt id in ``pending_request_info_events``.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import sqlite3
import time
import uuid
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent_framework import HistoryProvider, Message
from agent_framework._sessions import filter_new_messages
from agent_framework._workflows._checkpoint import CheckpointID, WorkflowCheckpoint
from agent_framework.ag_ui import AGUIThreadSnapshot
from agent_framework.exceptions import WorkflowCheckpointException

logger = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


_SCHEMA = """
CREATE TABLE IF NOT EXISTS thread_snapshots (
    scope TEXT NOT NULL,
    thread_id TEXT NOT NULL,
    snapshot_json TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (scope, thread_id)
);

CREATE TABLE IF NOT EXISTS history_messages (
    session_id TEXT NOT NULL,
    message_id TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    message_json TEXT NOT NULL,
    PRIMARY KEY (session_id, message_id)
);

CREATE TABLE IF NOT EXISTS checkpoints (
    checkpoint_id TEXT PRIMARY KEY,
    workflow_name TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    checkpoint_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_checkpoints_workflow_name
    ON checkpoints (workflow_name, timestamp);
"""


class DemoSqliteStore:
    """Shared SQLite connection helper for the handoff2 demo."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        with self._connect() as connection:
            connection.executescript(_SCHEMA)
            self._migrate(connection)
            connection.commit()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        return connection

    @staticmethod
    def _migrate(connection: sqlite3.Connection) -> None:
        """Drop legacy shapes from earlier handoff2 revisions."""

        history_cols = {row["name"] for row in connection.execute("PRAGMA table_info(history_messages)").fetchall()}
        if "history_key" in history_cols and "session_id" not in history_cols:
            connection.execute("DROP TABLE history_messages")
            connection.execute(
                """
                CREATE TABLE history_messages (
                    session_id TEXT NOT NULL,
                    message_id TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    message_json TEXT NOT NULL,
                    PRIMARY KEY (session_id, message_id)
                )
                """
            )

        tables = {
            str(row["name"])
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        if "thread_sessions" in tables:
            connection.execute("DROP TABLE thread_sessions")


class SqliteAGUIThreadSnapshotStore:
    """Durable latest AG-UI Thread Snapshot store (UI authority)."""

    def __init__(self, store: DemoSqliteStore) -> None:
        self._store = store

    @staticmethod
    def _snapshot_to_dict(snapshot: AGUIThreadSnapshot) -> dict[str, Any]:
        return {
            "messages": copy.deepcopy(snapshot.messages),
            "state": copy.deepcopy(snapshot.state),
            "interrupt": copy.deepcopy(snapshot.interrupt),
            "session_state": copy.deepcopy(snapshot.session_state),
        }

    @staticmethod
    def _snapshot_from_dict(payload: dict[str, Any]) -> AGUIThreadSnapshot:
        return AGUIThreadSnapshot(
            messages=list(payload.get("messages") or []),
            state=payload.get("state"),
            interrupt=payload.get("interrupt"),
            session_state=payload.get("session_state"),
        )

    async def save(self, *, scope: str, thread_id: str, snapshot: AGUIThreadSnapshot) -> None:
        payload = json.dumps(self._snapshot_to_dict(snapshot), ensure_ascii=False)
        async with self._store._lock:

            def _write() -> None:
                with self._store._connect() as connection:
                    connection.execute(
                        """
                        INSERT INTO thread_snapshots (scope, thread_id, snapshot_json, updated_at)
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT(scope, thread_id) DO UPDATE SET
                            snapshot_json = excluded.snapshot_json,
                            updated_at = excluded.updated_at
                        """,
                        (scope, thread_id, payload, _utc_now()),
                    )
                    connection.commit()

            await asyncio.to_thread(_write)

    async def get(self, *, scope: str, thread_id: str) -> AGUIThreadSnapshot | None:
        async with self._store._lock:

            def _read() -> AGUIThreadSnapshot | None:
                with self._store._connect() as connection:
                    row = connection.execute(
                        """
                        SELECT snapshot_json FROM thread_snapshots
                        WHERE scope = ? AND thread_id = ?
                        """,
                        (scope, thread_id),
                    ).fetchone()
                if row is None:
                    return None
                return self._snapshot_from_dict(json.loads(row["snapshot_json"]))

            return await asyncio.to_thread(_read)

    async def delete(self, *, scope: str, thread_id: str) -> bool:
        async with self._store._lock:

            def _delete() -> bool:
                with self._store._connect() as connection:
                    cursor = connection.execute(
                        "DELETE FROM thread_snapshots WHERE scope = ? AND thread_id = ?",
                        (scope, thread_id),
                    )
                    connection.commit()
                    return cursor.rowcount > 0

            return await asyncio.to_thread(_delete)

    async def clear(self, *, scope: str | None = None) -> None:
        async with self._store._lock:

            def _clear() -> None:
                with self._store._connect() as connection:
                    if scope is None:
                        connection.execute("DELETE FROM thread_snapshots")
                    else:
                        connection.execute("DELETE FROM thread_snapshots WHERE scope = ?", (scope,))
                    connection.commit()

            await asyncio.to_thread(_clear)


class SqliteCheckpointStorage:
    """SQLite workflow checkpoint storage (resume authority)."""

    def __init__(self, store: DemoSqliteStore, *, allowed_checkpoint_types: list[str] | None = None) -> None:
        self._store = store
        self._allowed_types = frozenset(allowed_checkpoint_types or [])

    async def save(self, checkpoint: WorkflowCheckpoint) -> CheckpointID:
        from agent_framework._workflows._checkpoint_encoding import encode_checkpoint_value

        encoded = encode_checkpoint_value(checkpoint.to_dict())
        payload = json.dumps(encoded, ensure_ascii=False)
        async with self._store._lock:

            def _write() -> None:
                with self._store._connect() as connection:
                    connection.execute(
                        """
                        INSERT INTO checkpoints (checkpoint_id, workflow_name, timestamp, checkpoint_json)
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT(checkpoint_id) DO UPDATE SET
                            workflow_name = excluded.workflow_name,
                            timestamp = excluded.timestamp,
                            checkpoint_json = excluded.checkpoint_json
                        """,
                        (checkpoint.checkpoint_id, checkpoint.workflow_name, checkpoint.timestamp, payload),
                    )
                    connection.commit()

            await asyncio.to_thread(_write)
        logger.debug("Saved checkpoint %s to sqlite", checkpoint.checkpoint_id)
        return checkpoint.checkpoint_id

    async def load(self, checkpoint_id: CheckpointID) -> WorkflowCheckpoint:
        from agent_framework._workflows._checkpoint_encoding import decode_checkpoint_value

        async with self._store._lock:

            def _read() -> dict[str, Any]:
                with self._store._connect() as connection:
                    row = connection.execute(
                        "SELECT checkpoint_json FROM checkpoints WHERE checkpoint_id = ?",
                        (checkpoint_id,),
                    ).fetchone()
                if row is None:
                    raise WorkflowCheckpointException(f"No checkpoint found with ID {checkpoint_id}")
                return json.loads(row["checkpoint_json"])

            encoded = await asyncio.to_thread(_read)

        decoded = decode_checkpoint_value(encoded, allowed_types=self._allowed_types)
        return WorkflowCheckpoint.from_dict(decoded)

    async def list_checkpoints(self, *, workflow_name: str) -> list[WorkflowCheckpoint]:
        from agent_framework._workflows._checkpoint_encoding import decode_checkpoint_value

        async with self._store._lock:

            def _list() -> list[dict[str, Any]]:
                with self._store._connect() as connection:
                    rows = connection.execute(
                        """
                        SELECT checkpoint_json FROM checkpoints
                        WHERE workflow_name = ?
                        ORDER BY timestamp ASC
                        """,
                        (workflow_name,),
                    ).fetchall()
                return [json.loads(row["checkpoint_json"]) for row in rows]

            encoded_rows = await asyncio.to_thread(_list)

        return [
            WorkflowCheckpoint.from_dict(decode_checkpoint_value(item, allowed_types=self._allowed_types))
            for item in encoded_rows
        ]

    async def delete(self, checkpoint_id: CheckpointID) -> bool:
        async with self._store._lock:

            def _delete() -> bool:
                with self._store._connect() as connection:
                    cursor = connection.execute(
                        "DELETE FROM checkpoints WHERE checkpoint_id = ?",
                        (checkpoint_id,),
                    )
                    connection.commit()
                    return cursor.rowcount > 0

            return await asyncio.to_thread(_delete)

    async def get_latest(self, *, workflow_name: str) -> WorkflowCheckpoint | None:
        checkpoints = await self.list_checkpoints(workflow_name=workflow_name)
        if not checkpoints:
            return None
        return max(checkpoints, key=lambda item: datetime.fromisoformat(item.timestamp))

    async def list_checkpoint_ids(self, *, workflow_name: str) -> list[CheckpointID]:
        return [checkpoint.checkpoint_id for checkpoint in await self.list_checkpoints(workflow_name=workflow_name)]


def _ensure_message_id(message: Message) -> str:
    message_id = getattr(message, "message_id", None) or getattr(message, "id", None)
    if isinstance(message_id, str) and message_id:
        return message_id
    new_id = str(uuid.uuid4())
    if hasattr(message, "message_id"):
        message.message_id = new_id  # type: ignore[attr-defined]
    return new_id


def _next_created_at_ms(sequence_seconds: list[int]) -> int:
    created_at = sequence_seconds[0] * 1000
    sequence_seconds[0] += 1
    return created_at


class SqliteHistoryProvider(HistoryProvider):
    """Durable HistoryProvider keyed by AgentSession.session_id (same shape as platform store).

    Checkpoint restore brings back the AgentSession (including session_id), so history
    follows the active run automatically.
    """

    DEFAULT_SOURCE_ID = "session_history"

    def __init__(
        self,
        store: DemoSqliteStore,
        *,
        source_id: str = DEFAULT_SOURCE_ID,
        load_messages: bool = True,
        store_inputs: bool = True,
        store_outputs: bool = True,
    ) -> None:
        super().__init__(
            source_id,
            load_messages=load_messages,
            store_inputs=store_inputs,
            store_outputs=store_outputs,
            store_context_messages=False,
        )
        self._store = store

    async def get_messages(
        self,
        session_id: str | None,
        *,
        state: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Message]:
        del kwargs
        if not session_id or state is None:
            return []
        stored = state.get("messages")
        if isinstance(stored, list) and stored:
            return list(stored)

        async with self._store._lock:

            def _read() -> list[Message]:
                with self._store._connect() as connection:
                    rows = connection.execute(
                        """
                        SELECT message_json FROM history_messages
                        WHERE session_id = ?
                        ORDER BY created_at ASC, message_id ASC
                        """,
                        (session_id,),
                    ).fetchall()
                return [Message.from_dict(json.loads(row["message_json"])) for row in rows]

            loaded = await asyncio.to_thread(_read)
        state["messages"] = list(loaded)
        return list(loaded)

    async def save_messages(
        self,
        session_id: str | None,
        messages: Sequence[Message],
        *,
        state: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        del kwargs
        if not session_id or state is None or not messages:
            return

        existing = state.get("messages")
        if not isinstance(existing, list):
            existing = []
            state["messages"] = existing
        new_messages = filter_new_messages(existing, messages)
        if new_messages:
            existing.extend(new_messages)

        rows: list[tuple[str, str, int]] = []
        sequence_seconds = [int(time.time())]
        for message in existing:
            message_id = _ensure_message_id(message)
            rows.append(
                (
                    message_id,
                    json.dumps(message.to_dict(), ensure_ascii=False),
                    _next_created_at_ms(sequence_seconds),
                )
            )

        async with self._store._lock:

            def _write() -> None:
                with self._store._connect() as connection:
                    for message_id, payload, created_at in rows:
                        connection.execute(
                            """
                            INSERT INTO history_messages
                                (session_id, message_id, created_at, message_json)
                            VALUES (?, ?, ?, ?)
                            ON CONFLICT(session_id, message_id) DO UPDATE SET
                                message_json = excluded.message_json
                            """,
                            (session_id, message_id, created_at, payload),
                        )
                    connection.commit()

            await asyncio.to_thread(_write)
