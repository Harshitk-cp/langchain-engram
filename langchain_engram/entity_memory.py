"""Engram entity memory for LangChain."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ._compat import BaseMemory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import PrivateAttr

from engram import Engram


class EngramEntityMemory(BaseMemory):
    """Memory that extracts structured entities and facts from conversations
    using Engram's LLM extraction pipeline.

    Unlike :class:`EngramChatMemory` (which stores raw conversation turns),
    ``EngramEntityMemory`` sends each exchange to Engram's extraction endpoint,
    which uses an LLM to identify typed memories — preferences, facts, decisions,
    constraints — and stores them with confidence scores.

    Concretely:

    - ``"I always work in dark mode"`` → stored as ``preference``, confidence 0.95
    - ``"I'm a senior Go engineer"`` → stored as ``fact``, confidence 0.92
    - ``"We can't use Python 2"`` → stored as ``constraint``, confidence 0.98
    - If a later message contradicts a stored belief, Engram's belief-update
      logic applies automatically — no extra code needed.

    The retrieved context on the next turn is structured facts, not raw dialogue,
    making it cleaner to inject into prompts.

    Auth:
        Pass ``api_key`` directly, or set ``ENGRAM_API_KEY`` environment variable.
        Keys are prefixed ``mk_`` (master) or ``rk_`` (restricted read+write scope).
        Pass ``base_url`` directly, or set ``ENGRAM_BASE_URL``.

    Args:
        agent_id: Engram agent ID that owns these memories.
        api_key: Engram API key. Falls back to ``ENGRAM_API_KEY`` env var.
        base_url: Engram server URL. Falls back to ``ENGRAM_BASE_URL`` env var.
        memory_key: Key used to inject entities into chain inputs. Default: ``"entities"``.
        input_key: Key in ``inputs`` holding the human message. Auto-detected when
            there is only one non-system key.
        output_key: Key in ``outputs`` holding the AI response. Auto-detected when
            there is only one key.
        return_messages: Return ``List[BaseMessage]`` instead of a formatted string.
        top_k: Max memories to retrieve per turn. Default: ``10``.
        min_confidence: Confidence floor (0–1). Filters out uncertain memories.

    Example::

        from langchain_engram import EngramEntityMemory
        from langchain.chains import LLMChain
        from langchain_core.prompts import PromptTemplate
        from langchain_openai import ChatOpenAI

        memory = EngramEntityMemory(
            agent_id="agent-uuid-here",
            api_key="mk_...",       # or set ENGRAM_API_KEY
            base_url="http://localhost:8080",  # or set ENGRAM_BASE_URL
        )

        prompt = PromptTemplate(
            input_variables=["entities", "input"],
            template=(
                "You are a helpful assistant.\\n"
                "What you know about this user:\\n{entities}\\n\\n"
                "Human: {input}\\nAI:"
            ),
        )

        chain = LLMChain(llm=ChatOpenAI(), prompt=prompt, memory=memory)
        chain.predict(input="I work in Go and I need a REST API for a todo app")
        chain.predict(input="What stack do you recommend for my project?")
        # → LLM sees structured facts: Go engineer, REST API project
    """

    agent_id: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    memory_key: str = "entities"
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
        """Recall structured entity facts relevant to the current input."""
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
        """Extract and persist structured entities from the conversation turn.

        Sends the human + AI exchange to Engram's LLM extraction endpoint.
        Engram identifies typed memories (preferences, facts, decisions,
        constraints) and stores them automatically (``auto_store=True``).
        Contradictions with existing memories trigger Engram's belief-update
        logic without any extra work here.
        """
        input_key = self._resolve_input_key(inputs)
        output_key = self._resolve_output_key(outputs)

        human_text = str(inputs.get(input_key, "")).strip()
        ai_text = str(outputs.get(output_key, "")).strip()

        conversation = []
        if human_text:
            conversation.append({"role": "user", "content": human_text})
        if ai_text:
            conversation.append({"role": "assistant", "content": ai_text})

        if not conversation:
            return

        self._client.memories.extract(
            agent_id=self.agent_id,
            conversation=conversation,
            auto_store=True,
        )

    def clear(self) -> None:
        """Clear the working memory session for this agent.

        Clears the active working-memory session without deleting stored
        long-term memories.
        """
        self._client.cognitive.clear_session(self.agent_id)

    # --- Helpers ---

    def _resolve_input_key(self, inputs: Dict[str, Any]) -> str:
        if self.input_key:
            return self.input_key
        keys = [k for k in inputs if k not in {"stop", "callbacks", "tags", "metadata"}]
        if len(keys) == 1:
            return keys[0]
        raise ValueError(
            f"EngramEntityMemory found multiple input keys: {list(inputs.keys())}. "
            "Set input_key explicitly on EngramEntityMemory."
        )

    def _resolve_output_key(self, outputs: Dict[str, Any]) -> str:
        if self.output_key:
            return self.output_key
        keys = list(outputs.keys())
        if len(keys) == 1:
            return keys[0]
        raise ValueError(
            f"EngramEntityMemory found multiple output keys: {list(outputs.keys())}. "
            "Set output_key explicitly on EngramEntityMemory."
        )

    def _as_string(self, memories: list) -> str:
        if not memories:
            return ""
        lines = [
            f"- {mem.content} [{mem.type.value}, {int(mem.confidence * 100)}% confident]"
            for mem in memories
        ]
        return "Known entities and facts:\n" + "\n".join(lines)

    def _as_messages(self, memories: list) -> List[BaseMessage]:
        # Entity facts are injected as HumanMessage context so the LLM treats
        # them as established background, not AI assertions.
        return [HumanMessage(content=mem.content) for mem in memories]
