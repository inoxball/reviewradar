"""Column types that behave identically on Postgres (production) and SQLite (tests, local)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pgvector.sqlalchemy import VECTOR
from sqlalchemy import JSON, BigInteger, DateTime, Dialect, Integer, LargeBinary
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.types import TypeDecorator, TypeEngine

BigIntPrimaryKey = BigInteger().with_variant(Integer(), "sqlite")
"""64-bit keys on Postgres; SQLite only auto-increments ``INTEGER PRIMARY KEY`` columns."""

JSONPayload = JSON().with_variant(JSONB(), "postgresql")
"""Binary, indexable JSON on Postgres; plain JSON text elsewhere."""


class UTCDateTime(TypeDecorator[datetime]):
    """Timezone-aware UTC timestamps on every backend.

    Postgres stores ``timestamptz`` natively, but SQLite has no timezone support and
    returns naive values. Normalizing at the boundary keeps datetime comparisons in
    application code correct regardless of the database in use.
    """

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError("naive datetimes are not allowed; pass a timezone-aware value")
        return value.astimezone(UTC)

    def process_result_value(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


class EmbeddingVector(TypeDecorator[NDArray[np.float32]]):
    """Dense float32 vectors: pgvector ``vector`` on Postgres, packed float32 bytes elsewhere.

    The column is deliberately dimension-less so several embedding models can share one
    table; approximate search uses a per-model partial HNSW index that casts to the
    model's dimension (see migration ``0002``).
    """

    impl = LargeBinary
    cache_ok = True

    def load_dialect_impl(self, dialect: Dialect) -> TypeEngine[Any]:
        if dialect.name == "postgresql":
            return dialect.type_descriptor(VECTOR())
        return dialect.type_descriptor(LargeBinary())

    def process_bind_param(self, value: NDArray[np.float32] | None, dialect: Dialect) -> Any:
        if value is None:
            return None
        vector = np.asarray(value, dtype=np.float32)
        if vector.ndim != 1:
            raise ValueError(f"embeddings must be one-dimensional, got shape {vector.shape}")
        return vector if dialect.name == "postgresql" else vector.tobytes()

    def process_result_value(self, value: Any, dialect: Dialect) -> NDArray[np.float32] | None:
        if value is None:
            return None
        if isinstance(value, bytes | bytearray | memoryview):
            return np.frombuffer(value, dtype=np.float32).copy()
        return np.asarray(value, dtype=np.float32)
