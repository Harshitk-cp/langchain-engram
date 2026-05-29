"""Engram retriever for LangChain."""

from __future__ import annotations

from typing import Any, List, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import PrivateAttr

from engram import Engram


class EngramRetriever(BaseRetriever):
    """LangChain retriever backed by Engram hybrid vector + graph recall.

    Retrieves memories for an agent using Engram's hybrid recall pipeline:
    vector similarity combined with knowledge-graph traversal. Each memory
    is returned as a LangChain ``Document`` with full metadata including
    confidence score, memory tier, and per-channel relevance scores.

    Only memories at or above ``min_confidence`` are returned — uncertain or
    decayed memories are filtered out automatically, so the retriever surfaces
    only what the agent is actually confident about.

    Auth:
        Pass ``api_key`` directly, or set the ``ENGRAM_API_KEY`` environment variable.
        Keys are prefixed ``mk_`` (master) or ``rk_`` (restricted read+write scope).
        Pass ``base_url`` directly, or set ``ENGRAM_BASE_URL``.

    Args:
        agent_id: Engram agent ID to retrieve memories from.
        api_key: Engram API key. Falls back to ``ENGRAM_API_KEY`` env var.
        base_url: Engram server URL. Falls back to ``ENGRAM_BASE_URL`` env var.
        top_k: Max memories to return. Default: ``10``.
        min_confidence: Confidence floor (0–1). Memories below this are excluded.
        memory_type: Filter by type — ``"fact"``, ``"preference"``, ``"decision"``,
            or ``"constraint"``. Returns all types if not set.
        graph_weight: Graph/vector blend (0–1). Higher values weight knowledge-graph
            traversal more heavily. Server default is 0.4 / 0.6 vector.

    Document metadata fields:
        - ``memory_id``: UUID of the memory
        - ``agent_id``: Owning agent UUID
        - ``memory_type``: ``"fact"`` | ``"preference"`` | ``"decision"`` | ``"constraint"``
        - ``confidence``: Float 0–1
        - ``tier``: ``"hot"`` | ``"warm"`` | ``"cold"``
        - ``source``: e.g. ``"conversation_human"`` | ``"conversation_ai"``
        - ``score``: Combined recall score
        - ``vector_score``: Vector similarity component
        - ``graph_score``: Graph traversal component
        - ``created_at``: ISO 8601 timestamp

    Example::

        from langchain_engram import EngramRetriever
        from langchain.chains import RetrievalQA
        from langchain_openai import ChatOpenAI

        retriever = EngramRetriever(
            agent_id="agent-uuid-here",
            api_key="mk_...",
            base_url="http://localhost:8080",
            top_k=5,
            min_confidence=0.6,  # only confident memories
        )

        qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(), retriever=retriever)
        answer = qa.invoke({"query": "What are the user's display preferences?"})

    As a standalone retriever::

        docs = retriever.invoke("display preferences")
        for doc in docs:
            print(doc.page_content, doc.metadata["confidence"])
    """

    agent_id: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    top_k: int = 10
    min_confidence: Optional[float] = None
    memory_type: Optional[str] = None
    graph_weight: Optional[float] = None

    _client: Engram = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        self._client = Engram(base_url=self.base_url, api_key=self.api_key)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        result = self._client.memories.recall(
            agent_id=self.agent_id,
            query=query,
            top_k=self.top_k,
            type=self.memory_type,
            min_confidence=self.min_confidence,
            graph_weight=self.graph_weight,
        )

        documents = []
        for mem in result.memories:
            metadata: dict = {
                "memory_id": mem.id,
                "agent_id": mem.agent_id,
                "memory_type": mem.type.value if mem.type else None,
                "confidence": mem.confidence,
                "source": mem.source,
                "created_at": mem.created_at.isoformat() if mem.created_at else None,
            }
            if mem.tier:
                metadata["tier"] = mem.tier.value
            if mem.score is not None:
                metadata["score"] = mem.score
            if mem.vector_score is not None:
                metadata["vector_score"] = mem.vector_score
            if mem.graph_score is not None:
                metadata["graph_score"] = mem.graph_score

            documents.append(Document(page_content=mem.content, metadata=metadata))

        return documents
