"""Engram chat memory for LangChain."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_core.memory import BaseMemory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import PrivateAttr

from engram import Engram


class EngramChatMemory(BaseMemory):
    """Persistent, confidence-aware chat memory backed by Engram.

    Unlike LangChain's built-in memory classes, EngramChatMemory persists
    conversation turns as Engram memories and retrieves *semantically relevant*
    context on each turn — not a flat buffer of recent messages.

    Each stored memory goes through Engram's full cognitive model: LLM-based type
    classification, confidence scoring, contradiction detection, and decay lifecycle.
    Hot-tier memories (high confidence, recently used) are surfaced first.

    Auth:
        Pass ``api_key`` directly, or set the ``ENGRAM_API_KEY`` environment variable.
        Keys are prefixed ``mk_`` (master) or ``rk_`` (restricted read+write scope).
        Pass ``base_url`` directly, or set ``ENGRAM_BASE_URL``.

    Args:
        agent_id: Engram agent ID that owns these memories.
        api_key: Engram API key. Falls back to ``ENGRAM_API_KEY`` env var.
        base_url: Engram server URL. Falls back to ``ENGRAM_BASE_URL`` env var.
        memory_key: Key used to inject memories into chain inputs. Default: ``"history"``.
        input_key: Key in ``inputs`` holding the human message. Auto-detected when
            there is only one non-system key.
        output_key: Key in ``outputs`` holding the AI response. Auto-detected when
            there is only one key.
        return_messages: Return ``List[BaseMessage]`` instead of a formatted string.
            Human-sourced memories become ``HumanMessage``; AI-sourced become ``AIMessage``.
        top_k: Max memories to retrieve per turn. Default: ``10``.
        min_confidence: Confidence floor (0–1). Filters out uncertain memories.

    Example::

        from langchain_engram import EngramChatMemory
        from langchain.chains import ConversationChain
        from langchain_openai import ChatOpenAI

        memory = EngramChatMemory(
            agent_id="agent-uuid-here",
            api_key="mk_...",       # or set ENGRAM_API_KEY
            base_url="http://localhost:8080",  # or set ENGRAM_BASE_URL
        )

        chain = ConversationChain(llm=ChatOpenAI(), memory=memory)
        chain.predict(input="I always prefer dark mode in my tools")
        chain.predict(input="What are my display preferences?")
        # → returns answer informed by the stored preference
    """

    agent_id: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    memory_key: str = "history"
    input_key: Optional[str] = None
    output_key: Optional[str] = None
    return_messages: bool = False
    top_k: int = 10
    min_confidence: Optional[float] = None

    _client: Engram = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        self._client = Engram(base_url=self.base_url, api_key=self.api_key)

    # --- LangChain BaseMemory interface ---

    @property
    def memory_variables(self) -> List[str]:
        return [self.memory_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Recall semantically relevant memories for the current input."""
        query = ""
        if inputs:
            key = self._resolve_input_key(inputs)
            query = str(inputs.get(key, ""))

        result = self._client.memories.recall(
            agent_id=self.agent_id,
            query=query or "general context",
            top_k=self.top_k,
            min_confidence=self.min_confidence,
        )

        if self.return_messages:
            return {self.memory_key: self._as_messages(result.memories)}

        return {self.memory_key: self._as_string(result.memories)}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        """Persist the conversation turn — human input and AI response — as Engram memories."""
        input_key = self._resolve_input_key(inputs)
        output_key = self._resolve_output_key(outputs)

        human_text = str(inputs.get(input_key, "")).strip()
        ai_text = str(outputs.get(output_key, "")).strip()

        if human_text:
            self._client.memories.store(
                agent_id=self.agent_id,
                content=human_text,
                source="conversation_human",
            )
        if ai_text:
            self._client.memories.store(
                agent_id=self.agent_id,
                content=ai_text,
                source="conversation_ai",
            )

    def clear(self) -> None:
        """Clear the working memory session for this agent.

        This clears the active working-memory session (context window) without
        deleting stored long-term memories. To permanently delete memories, use
        the Engram client directly.
        """
        self._client.cognitive.clear_session(self.agent_id)

    # --- Helpers ---

    def _resolve_input_key(self, inputs: Dict[str, Any]) -> str:
        if self.input_key:
            return self.input_key
        # Strip LangChain-internal keys that aren't part of user input
        keys = [k for k in inputs if k not in {"stop", "callbacks", "tags", "metadata"}]
        if len(keys) == 1:
            return keys[0]
        raise ValueError(
            f"EngramChatMemory found multiple input keys: {list(inputs.keys())}. "
            "Set input_key explicitly on EngramChatMemory."
        )

    def _resolve_output_key(self, outputs: Dict[str, Any]) -> str:
        if self.output_key:
            return self.output_key
        keys = list(outputs.keys())
        if len(keys) == 1:
            return keys[0]
        raise ValueError(
            f"EngramChatMemory found multiple output keys: {list(outputs.keys())}. "
            "Set output_key explicitly on EngramChatMemory."
        )

    def _as_string(self, memories: list) -> str:
        if not memories:
            return ""
        lines = [
            f"- {mem.content} [{int(mem.confidence * 100)}% confidence]"
            for mem in memories
        ]
        return "Relevant context from memory:\n" + "\n".join(lines)

    def _as_messages(self, memories: list) -> List[BaseMessage]:
        messages: List[BaseMessage] = []
        for mem in memories:
            if mem.source == "conversation_ai":
                messages.append(AIMessage(content=mem.content))
            else:
                messages.append(HumanMessage(content=mem.content))
        return messages
