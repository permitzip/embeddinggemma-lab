from __future__ import annotations

import math
import sqlite3
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class Hit:
    score: float
    modality: str
    path: str
    title: str
    excerpt: str
    chunk_index: int


def pack_f32(values: Sequence[float]) -> bytes:
    return struct.pack(f"<{len(values)}f", *[float(value) for value in values])


def unpack_f32(blob: bytes) -> list[float]:
    if len(blob) % 4 != 0:
        raise ValueError("embedding blob is not a float32 sequence")
    count = len(blob) // 4
    return list(struct.unpack(f"<{count}f", blob))


def project(values: Sequence[float], dim: int) -> list[float]:
    if dim < 1 or dim > len(values):
        raise ValueError(f"dimension {dim} is outside 1..{len(values)}")
    sliced = [float(value) for value in values[:dim]]
    norm = math.sqrt(sum(value * value for value in sliced))
    if norm == 0:
        return sliced
    return [value / norm for value in sliced]


def dot(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError(f"vector length {len(left)} does not match {len(right)}")
    return sum(a * b for a, b in zip(left, right))


class IndexStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta (
              key TEXT PRIMARY KEY,
              value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chunks (
              path TEXT NOT NULL,
              chunk_index INTEGER NOT NULL,
              modality TEXT NOT NULL,
              title TEXT NOT NULL,
              sha256 TEXT NOT NULL,
              mtime REAL NOT NULL,
              size_bytes INTEGER NOT NULL,
              excerpt TEXT NOT NULL,
              dim INTEGER NOT NULL,
              embedding BLOB NOT NULL,
              PRIMARY KEY (path, chunk_index)
            );
            """
        )

    def close(self) -> None:
        self.conn.close()

    def meta(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        if row is None:
            return None
        return str(row["value"])

    def set_meta(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO meta(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
        self.conn.commit()

    def prepare_for_model(self, model_id: str, version: str, dim: int) -> bool:
        """Wipe the index when the model or embedder version changes. Returns True if wiped."""
        current = (self.meta("model_id"), self.meta("embedder_version"), self.meta("dim"))
        wanted = (model_id, version, str(dim))
        if current == (None, None, None):
            self.set_meta("model_id", model_id)
            self.set_meta("embedder_version", version)
            self.set_meta("dim", str(dim))
            return False
        if current == wanted:
            return False
        self.conn.execute("DELETE FROM chunks")
        self.conn.commit()
        self.set_meta("model_id", model_id)
        self.set_meta("embedder_version", version)
        self.set_meta("dim", str(dim))
        return True

    def chunks_for(self, path: str) -> list[sqlite3.Row]:
        return list(
            self.conn.execute(
                "SELECT chunk_index, sha256 FROM chunks WHERE path = ? ORDER BY chunk_index",
                (path,),
            )
        )

    def is_current(self, path: str, digest: str, chunk_count: int) -> bool:
        rows = self.chunks_for(path)
        if len(rows) != chunk_count or chunk_count == 0:
            return False
        return all(str(row["sha256"]) == digest for row in rows)

    def append_chunks(self, path: str, rows: list[dict[str, object]], *, replace: bool) -> None:
        if replace:
            self.conn.execute("DELETE FROM chunks WHERE path = ?", (path,))
        if rows:
            self.conn.executemany(
                """
                INSERT INTO chunks (
                  path, chunk_index, modality, title, sha256, mtime, size_bytes, excerpt, dim, embedding
                ) VALUES (
                  :path, :chunk_index, :modality, :title, :sha256, :mtime, :size_bytes, :excerpt, :dim, :embedding
                )
                """,
                rows,
            )
        self.conn.commit()

    def replace_file(self, rows: list[dict[str, object]]) -> None:
        if not rows:
            return
        path = str(rows[0]["path"])
        self.conn.execute("DELETE FROM chunks WHERE path = ?", (path,))
        self.conn.executemany(
            """
            INSERT INTO chunks (
              path, chunk_index, modality, title, sha256, mtime, size_bytes, excerpt, dim, embedding
            ) VALUES (
              :path, :chunk_index, :modality, :title, :sha256, :mtime, :size_bytes, :excerpt, :dim, :embedding
            )
            """,
            rows,
        )
        self.conn.commit()

    def count(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()
        return int(row["n"])

    def list_rows(self) -> list[sqlite3.Row]:
        return list(
            self.conn.execute(
                "SELECT path, chunk_index, modality, title FROM chunks ORDER BY path, chunk_index"
            )
        )

    def search(
        self,
        query: Sequence[float],
        *,
        top: int,
        dim: int,
        modality: str | None = None,
    ) -> list[Hit]:
        projected = project(query, dim)
        hits: list[Hit] = []
        sql = "SELECT path, chunk_index, modality, title, excerpt, embedding FROM chunks"
        args: tuple[object, ...] = ()
        if modality is not None:
            sql += " WHERE modality = ?"
            args = (modality,)
        for row in self.conn.execute(sql, args):
            vector = unpack_f32(row["embedding"])
            score = dot(project(vector, dim), projected)
            hits.append(
                Hit(
                    score=score,
                    modality=str(row["modality"]),
                    path=str(row["path"]),
                    title=str(row["title"]),
                    excerpt=str(row["excerpt"]),
                    chunk_index=int(row["chunk_index"]),
                )
            )
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:top]
