"""Compatibility shim for BaseMemory across langchain-core versions.

langchain_core.memory was added in 0.3.3 and is marked for removal in 1.0.0.
Different versions of langchain / langsmith / langchain-core pull in different
ranges, so we can't rely on a single import path. If the module exists, use it
(so isinstance checks against LangChain's own BaseMemory still work). If not,
fall back to a minimal Pydantic implementation with the same interface.
"""

from __future__ import annotations

try:
    from langchain_core.memory import BaseMemory  # langchain-core 0.3.3 – 0.x
except ImportError:
    from abc import abstractmethod
    from typing import Any

    from pydantic import BaseModel, ConfigDict

    class BaseMemory(BaseModel):  # type: ignore[no-redef]
        """Minimal BaseMemory shim for langchain-core versions that don't ship it."""

        model_config = ConfigDict(arbitrary_types_allowed=True)

        @property
        @abstractmethod
        def memory_variables(self) -> list[str]: ...  # pragma: no cover

        @abstractmethod
        def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, Any]: ...  # pragma: no cover

        @abstractmethod
        def save_context(self, inputs: dict[str, Any], outputs: dict[str, Any]) -> None: ...  # pragma: no cover

        @abstractmethod
        def clear(self) -> None: ...  # pragma: no cover


__all__ = ["BaseMemory"]
