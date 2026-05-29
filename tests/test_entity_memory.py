"""Tests for EngramEntityMemory."""

from __future__ import annotations

import httpx
import pytest
import respx
from langchain_core.messages import HumanMessage

from langchain_engram import EngramEntityMemory

BASE_URL = "http://test.engram.local"
API_KEY = "mk_testkey123"
AGENT_ID = "agent-abc-123"

_EXTRACT_RESPONSE = {
    "extracted": [
        {
            "type": "preference",
            "content": "User prefers dark mode",
            "confidence": 0.95,
            "evidence_type": "explicit_statement",
            "id": "mem-1",
        },
        {
            "type": "fact",
            "content": "User is a senior Go engineer",
            "confidence": 0.90,
            "evidence_type": "explicit_statement",
            "id": "mem-2",
        },
    ],
    "count": 2,
}

_EXTRACT_EMPTY = {"extracted": [], "count": 0}

_RECALL_RESPONSE = {
    "memories": [
        {
            "id": "mem-1",
            "agent_id": AGENT_ID,
            "tenant_id": "tenant-1",
            "type": "preference",
            "content": "User prefers dark mode",
            "confidence": 0.95,
            "source": "conversation_human",
            "tier": "hot",
        },
        {
            "id": "mem-2",
            "agent_id": AGENT_ID,
            "tenant_id": "tenant-1",
            "type": "fact",
            "content": "User is a senior Go engineer",
            "confidence": 0.90,
            "source": "conversation_human",
            "tier": "hot",
        },
    ],
    "query": "engineer preferences",
    "count": 2,
}

_RECALL_EMPTY = {"memories": [], "query": "", "count": 0}


# ---------------------------------------------------------------------------
# save_context — uses extract endpoint, not store
# ---------------------------------------------------------------------------


@respx.mock
def test_save_context_calls_extract_not_store():
    extract_route = respx.post(f"{BASE_URL}/v1/memories/extract").mock(
        return_value=httpx.Response(200, json=_EXTRACT_RESPONSE)
    )
    store_route = respx.post(f"{BASE_URL}/v1/memories/").mock(
        return_value=httpx.Response(201, json={})
    )

    memory = EngramEntityMemory(agent_id=AGENT_ID, api_key=API_KEY, base_url=BASE_URL)
    memory.save_context(
        inputs={"input": "I always prefer dark mode and I'm a senior Go engineer"},
        outputs={"output": "Got it, I'll keep those in mind."},
    )

    assert extract_route.called
    assert not store_route.called


@respx.mock
def test_save_context_sends_both_turns_as_conversation():
    route = respx.post(f"{BASE_URL}/v1/memories/extract").mock(
        return_value=httpx.Response(200, json=_EXTRACT_RESPONSE)
    )

    memory = EngramEntityMemory(agent_id=AGENT_ID, api_key=API_KEY, base_url=BASE_URL)
    memory.save_context(
        inputs={"input": "I prefer Go over Python"},
        outputs={"output": "Noted, I'll recommend Go-based solutions."},
    )

    assert route.called
    body = route.calls[0].request.content
    import json
    payload = json.loads(body)
    conversation = payload["conversation"]
    assert len(conversation) == 2
    assert conversation[0]["role"] == "user"
    assert conversation[1]["role"] == "assistant"
    assert payload["auto_store"] is True


@respx.mock
def test_save_context_skips_extract_when_both_empty():
    route = respx.post(f"{BASE_URL}/v1/memories/extract").mock(
        return_value=httpx.Response(200, json=_EXTRACT_EMPTY)
    )

    memory = EngramEntityMemory(agent_id=AGENT_ID, api_key=API_KEY, base_url=BASE_URL)
    memory.save_context(
        inputs={"input": "   "},
        outputs={"output": "  "},
    )

    assert not route.called


@respx.mock
def test_save_context_sends_only_human_turn_when_ai_empty():
    route = respx.post(f"{BASE_URL}/v1/memories/extract").mock(
        return_value=httpx.Response(200, json=_EXTRACT_EMPTY)
    )

    memory = EngramEntityMemory(agent_id=AGENT_ID, api_key=API_KEY, base_url=BASE_URL)
    memory.save_context(
        inputs={"input": "I prefer TypeScript"},
        outputs={"output": ""},
    )

    assert route.called
    import json
    payload = json.loads(route.calls[0].request.content)
    assert len(payload["conversation"]) == 1
    assert payload["conversation"][0]["role"] == "user"


# ---------------------------------------------------------------------------
# load_memory_variables — uses recall, formats as entities
# ---------------------------------------------------------------------------


@respx.mock
def test_load_returns_entity_formatted_string():
    respx.get(f"{BASE_URL}/v1/memories/recall").mock(
        return_value=httpx.Response(200, json=_RECALL_RESPONSE)
    )

    memory = EngramEntityMemory(agent_id=AGENT_ID, api_key=API_KEY, base_url=BASE_URL)
    result = memory.load_memory_variables({"input": "what stack should I use?"})

    assert "entities" in result
    text = result["entities"]
    assert text.startswith("Known entities and facts:")
    assert "dark mode" in text
    assert "preference" in text
    assert "95%" in text
    assert "Go engineer" in text


@respx.mock
def test_load_empty_recall_returns_empty_string():
    respx.get(f"{BASE_URL}/v1/memories/recall").mock(
        return_value=httpx.Response(200, json=_RECALL_EMPTY)
    )

    memory = EngramEntityMemory(agent_id=AGENT_ID, api_key=API_KEY, base_url=BASE_URL)
    result = memory.load_memory_variables({"input": "test"})
    assert result["entities"] == ""


@respx.mock
def test_load_return_messages():
    respx.get(f"{BASE_URL}/v1/memories/recall").mock(
        return_value=httpx.Response(200, json=_RECALL_RESPONSE)
    )

    memory = EngramEntityMemory(
        agent_id=AGENT_ID, api_key=API_KEY, base_url=BASE_URL, return_messages=True
    )
    result = memory.load_memory_variables({"input": "test"})

    msgs = result["entities"]
    assert isinstance(msgs, list)
    assert all(isinstance(m, HumanMessage) for m in msgs)
    assert len(msgs) == 2


# ---------------------------------------------------------------------------
# memory_key and memory_variables
# ---------------------------------------------------------------------------


def test_default_memory_key_is_entities():
    memory = EngramEntityMemory(agent_id=AGENT_ID, api_key=API_KEY, base_url=BASE_URL)
    assert memory.memory_variables == ["entities"]
    assert memory.memory_key == "entities"


def test_custom_memory_key():
    memory = EngramEntityMemory(
        agent_id=AGENT_ID, api_key=API_KEY, base_url=BASE_URL, memory_key="known_facts"
    )
    assert memory.memory_variables == ["known_facts"]


# ---------------------------------------------------------------------------
# clear
# ---------------------------------------------------------------------------


@respx.mock
def test_clear_calls_cognitive_session_delete():
    route = respx.delete(f"{BASE_URL}/v1/cognitive/session").mock(
        return_value=httpx.Response(204)
    )

    memory = EngramEntityMemory(agent_id=AGENT_ID, api_key=API_KEY, base_url=BASE_URL)
    memory.clear()

    assert route.called


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


@respx.mock
def test_bearer_token_sent_in_extract_request():
    route = respx.post(f"{BASE_URL}/v1/memories/extract").mock(
        return_value=httpx.Response(200, json=_EXTRACT_EMPTY)
    )

    memory = EngramEntityMemory(agent_id=AGENT_ID, api_key=API_KEY, base_url=BASE_URL)
    memory.save_context(
        inputs={"input": "I like Python"},
        outputs={"output": "Great."},
    )

    auth = route.calls[0].request.headers.get("authorization", "")
    assert auth == f"Bearer {API_KEY}"


@respx.mock
def test_api_key_env_fallback(monkeypatch):
    monkeypatch.setenv("ENGRAM_API_KEY", API_KEY)
    monkeypatch.setenv("ENGRAM_BASE_URL", BASE_URL)

    respx.get(f"{BASE_URL}/v1/memories/recall").mock(
        return_value=httpx.Response(200, json=_RECALL_EMPTY)
    )

    memory = EngramEntityMemory(agent_id=AGENT_ID)
    result = memory.load_memory_variables({"input": "test"})
    assert result["entities"] == ""
