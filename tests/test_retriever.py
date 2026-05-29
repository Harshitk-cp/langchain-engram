"""Tests for EngramRetriever."""

from __future__ import annotations

import httpx
import pytest
import respx
from langchain_core.documents import Document

from langchain_engram import EngramRetriever

BASE_URL = "http://test.engram.local"
API_KEY = "mk_testkey123"
AGENT_ID = "agent-abc-123"

_RECALL_RESPONSE = {
    "memories": [
        {
            "id": "mem-1",
            "agent_id": AGENT_ID,
            "tenant_id": "tenant-1",
            "type": "preference",
            "content": "User prefers dark mode",
            "confidence": 0.92,
            "source": "conversation_human",
            "tier": "hot",
            "score": 0.88,
            "vector_score": 0.85,
            "graph_score": 0.90,
        },
        {
            "id": "mem-2",
            "agent_id": AGENT_ID,
            "tenant_id": "tenant-1",
            "type": "fact",
            "content": "User uses VS Code as primary editor",
            "confidence": 0.87,
            "source": "conversation_human",
            "tier": "warm",
            "score": 0.72,
        },
    ],
    "query": "display preferences",
    "count": 2,
}

_RECALL_EMPTY = {"memories": [], "query": "", "count": 0}


# ---------------------------------------------------------------------------
# Basic retrieval
# ---------------------------------------------------------------------------


@respx.mock
def test_invoke_returns_documents():
    respx.get(f"{BASE_URL}/v1/memories/recall").mock(
        return_value=httpx.Response(200, json=_RECALL_RESPONSE)
    )

    retriever = EngramRetriever(agent_id=AGENT_ID, api_key=API_KEY, base_url=BASE_URL)
    docs = retriever.invoke("display preferences")

    assert len(docs) == 2
    assert all(isinstance(d, Document) for d in docs)
    assert docs[0].page_content == "User prefers dark mode"
    assert docs[1].page_content == "User uses VS Code as primary editor"


@respx.mock
def test_empty_recall_returns_empty_list():
    respx.get(f"{BASE_URL}/v1/memories/recall").mock(
        return_value=httpx.Response(200, json=_RECALL_EMPTY)
    )

    retriever = EngramRetriever(agent_id=AGENT_ID, api_key=API_KEY, base_url=BASE_URL)
    docs = retriever.invoke("anything")

    assert docs == []


# ---------------------------------------------------------------------------
# Document metadata
# ---------------------------------------------------------------------------


@respx.mock
def test_document_metadata_core_fields():
    respx.get(f"{BASE_URL}/v1/memories/recall").mock(
        return_value=httpx.Response(200, json=_RECALL_RESPONSE)
    )

    retriever = EngramRetriever(agent_id=AGENT_ID, api_key=API_KEY, base_url=BASE_URL)
    docs = retriever.invoke("display preferences")
    meta = docs[0].metadata

    assert meta["memory_id"] == "mem-1"
    assert meta["agent_id"] == AGENT_ID
    assert meta["memory_type"] == "preference"
    assert meta["confidence"] == 0.92
    assert meta["tier"] == "hot"
    assert meta["source"] == "conversation_human"


@respx.mock
def test_document_metadata_score_fields():
    respx.get(f"{BASE_URL}/v1/memories/recall").mock(
        return_value=httpx.Response(200, json=_RECALL_RESPONSE)
    )

    retriever = EngramRetriever(agent_id=AGENT_ID, api_key=API_KEY, base_url=BASE_URL)
    docs = retriever.invoke("display preferences")
    meta = docs[0].metadata

    assert meta["score"] == 0.88
    assert meta["vector_score"] == 0.85
    assert meta["graph_score"] == 0.90


@respx.mock
def test_document_metadata_missing_optional_scores():
    # mem-2 has no vector_score or graph_score
    respx.get(f"{BASE_URL}/v1/memories/recall").mock(
        return_value=httpx.Response(200, json=_RECALL_RESPONSE)
    )

    retriever = EngramRetriever(agent_id=AGENT_ID, api_key=API_KEY, base_url=BASE_URL)
    docs = retriever.invoke("display preferences")
    meta = docs[1].metadata

    assert "vector_score" not in meta
    assert "graph_score" not in meta


# ---------------------------------------------------------------------------
# Query params are forwarded
# ---------------------------------------------------------------------------


@respx.mock
def test_min_confidence_forwarded():
    route = respx.get(f"{BASE_URL}/v1/memories/recall").mock(
        return_value=httpx.Response(200, json=_RECALL_EMPTY)
    )

    retriever = EngramRetriever(
        agent_id=AGENT_ID, api_key=API_KEY, base_url=BASE_URL, min_confidence=0.8
    )
    retriever.invoke("test")

    assert route.called
    assert "min_confidence=0.8" in str(route.calls[0].request.url)


@respx.mock
def test_memory_type_filter_forwarded():
    route = respx.get(f"{BASE_URL}/v1/memories/recall").mock(
        return_value=httpx.Response(200, json=_RECALL_EMPTY)
    )

    retriever = EngramRetriever(
        agent_id=AGENT_ID, api_key=API_KEY, base_url=BASE_URL, memory_type="preference"
    )
    retriever.invoke("test")

    assert "type=preference" in str(route.calls[0].request.url)


@respx.mock
def test_graph_weight_forwarded():
    route = respx.get(f"{BASE_URL}/v1/memories/recall").mock(
        return_value=httpx.Response(200, json=_RECALL_EMPTY)
    )

    retriever = EngramRetriever(
        agent_id=AGENT_ID, api_key=API_KEY, base_url=BASE_URL, graph_weight=0.7
    )
    retriever.invoke("test")

    assert "graph_weight=0.7" in str(route.calls[0].request.url)


@respx.mock
def test_top_k_forwarded():
    route = respx.get(f"{BASE_URL}/v1/memories/recall").mock(
        return_value=httpx.Response(200, json=_RECALL_EMPTY)
    )

    retriever = EngramRetriever(
        agent_id=AGENT_ID, api_key=API_KEY, base_url=BASE_URL, top_k=3
    )
    retriever.invoke("test")

    assert "top_k=3" in str(route.calls[0].request.url)


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


@respx.mock
def test_bearer_token_sent_in_request():
    route = respx.get(f"{BASE_URL}/v1/memories/recall").mock(
        return_value=httpx.Response(200, json=_RECALL_EMPTY)
    )

    retriever = EngramRetriever(agent_id=AGENT_ID, api_key=API_KEY, base_url=BASE_URL)
    retriever.invoke("test")

    auth_header = route.calls[0].request.headers.get("authorization", "")
    assert auth_header == f"Bearer {API_KEY}"


@respx.mock
def test_api_key_env_fallback(monkeypatch):
    monkeypatch.setenv("ENGRAM_API_KEY", API_KEY)
    monkeypatch.setenv("ENGRAM_BASE_URL", BASE_URL)

    respx.get(f"{BASE_URL}/v1/memories/recall").mock(
        return_value=httpx.Response(200, json=_RECALL_EMPTY)
    )

    retriever = EngramRetriever(agent_id=AGENT_ID)
    docs = retriever.invoke("test")
    assert docs == []
