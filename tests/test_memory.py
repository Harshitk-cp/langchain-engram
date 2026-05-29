"""Tests for EngramChatMemory."""

from __future__ import annotations

import httpx
import pytest
import respx
from langchain_core.messages import AIMessage, HumanMessage

from langchain_engram import EngramChatMemory

BASE_URL = "http://test.engram.local"
API_KEY = "mk_testkey123"
AGENT_ID = "agent-abc-123"

_MEM_STUB = {
    "id": "mem-1",
    "agent_id": AGENT_ID,
    "tenant_id": "tenant-1",
    "type": "preference",
    "content": "User prefers dark mode",
    "confidence": 0.92,
    "source": "conversation_human",
    "tier": "hot",
    "score": 0.88,
}

_RECALL_RESPONSE = {
    "memories": [_MEM_STUB],
    "query": "display preferences",
    "count": 1,
}

_RECALL_EMPTY = {"memories": [], "query": "", "count": 0}

_STORE_RESPONSE = {**_MEM_STUB, "id": "mem-2", "content": "I prefer dark mode"}


# ---------------------------------------------------------------------------
# load_memory_variables
# ---------------------------------------------------------------------------


@respx.mock
def test_load_returns_formatted_string():
    respx.get(f"{BASE_URL}/v1/memories/recall").mock(
        return_value=httpx.Response(200, json=_RECALL_RESPONSE)
    )

    memory = EngramChatMemory(agent_id=AGENT_ID, api_key=API_KEY, base_url=BASE_URL)
    result = memory.load_memory_variables({"input": "display preferences"})

    assert "history" in result
    assert "dark mode" in result["history"]
    assert "92%" in result["history"]
    assert result["history"].startswith("Relevant context from memory:")


@respx.mock
def test_load_empty_recall_returns_empty_string():
    respx.get(f"{BASE_URL}/v1/memories/recall").mock(
        return_value=httpx.Response(200, json=_RECALL_EMPTY)
    )

    memory = EngramChatMemory(agent_id=AGENT_ID, api_key=API_KEY, base_url=BASE_URL)
    result = memory.load_memory_variables({"input": "something"})

    assert result["history"] == ""


@respx.mock
def test_load_return_messages_human_source():
    respx.get(f"{BASE_URL}/v1/memories/recall").mock(
        return_value=httpx.Response(200, json=_RECALL_RESPONSE)
    )

    memory = EngramChatMemory(
        agent_id=AGENT_ID, api_key=API_KEY, base_url=BASE_URL, return_messages=True
    )
    result = memory.load_memory_variables({"input": "display preferences"})

    msgs = result["history"]
    assert isinstance(msgs, list)
    assert isinstance(msgs[0], HumanMessage)
    assert msgs[0].content == "User prefers dark mode"


@respx.mock
def test_load_return_messages_ai_source():
    ai_mem = {**_MEM_STUB, "source": "conversation_ai", "content": "Got it, dark mode saved."}
    respx.get(f"{BASE_URL}/v1/memories/recall").mock(
        return_value=httpx.Response(200, json={"memories": [ai_mem], "query": "", "count": 1})
    )

    memory = EngramChatMemory(
        agent_id=AGENT_ID, api_key=API_KEY, base_url=BASE_URL, return_messages=True
    )
    result = memory.load_memory_variables({"input": "test"})

    assert isinstance(result["history"][0], AIMessage)


@respx.mock
def test_custom_memory_key_in_output():
    respx.get(f"{BASE_URL}/v1/memories/recall").mock(
        return_value=httpx.Response(200, json=_RECALL_RESPONSE)
    )

    memory = EngramChatMemory(
        agent_id=AGENT_ID, api_key=API_KEY, base_url=BASE_URL, memory_key="chat_history"
    )
    result = memory.load_memory_variables({"input": "test"})

    assert "chat_history" in result
    assert "history" not in result


# ---------------------------------------------------------------------------
# save_context
# ---------------------------------------------------------------------------


@respx.mock
def test_save_context_stores_both_turns():
    store_route = respx.post(f"{BASE_URL}/v1/memories/").mock(
        return_value=httpx.Response(201, json=_STORE_RESPONSE)
    )

    memory = EngramChatMemory(agent_id=AGENT_ID, api_key=API_KEY, base_url=BASE_URL)
    memory.save_context(
        inputs={"input": "I prefer dark mode"},
        outputs={"output": "Got it, I'll remember that."},
    )

    assert store_route.call_count == 2


@respx.mock
def test_save_context_skips_empty_strings():
    store_route = respx.post(f"{BASE_URL}/v1/memories/").mock(
        return_value=httpx.Response(201, json=_STORE_RESPONSE)
    )

    memory = EngramChatMemory(agent_id=AGENT_ID, api_key=API_KEY, base_url=BASE_URL)
    memory.save_context(
        inputs={"input": "  "},   # whitespace only — should be skipped
        outputs={"output": "Sure thing."},
    )

    # Only the AI response should be stored
    assert store_route.call_count == 1


@respx.mock
def test_save_context_explicit_keys():
    store_route = respx.post(f"{BASE_URL}/v1/memories/").mock(
        return_value=httpx.Response(201, json=_STORE_RESPONSE)
    )

    memory = EngramChatMemory(
        agent_id=AGENT_ID,
        api_key=API_KEY,
        base_url=BASE_URL,
        input_key="question",
        output_key="answer",
    )
    memory.save_context(
        inputs={"question": "Prefer dark mode", "metadata": {"session": "1"}},
        outputs={"answer": "Noted.", "debug": "..."},
    )

    assert store_route.call_count == 2


# ---------------------------------------------------------------------------
# clear
# ---------------------------------------------------------------------------


@respx.mock
def test_clear_calls_cognitive_session_delete():
    clear_route = respx.delete(f"{BASE_URL}/v1/cognitive/session").mock(
        return_value=httpx.Response(204)
    )

    memory = EngramChatMemory(agent_id=AGENT_ID, api_key=API_KEY, base_url=BASE_URL)
    memory.clear()

    assert clear_route.called


# ---------------------------------------------------------------------------
# Auth / env var fallback
# ---------------------------------------------------------------------------


@respx.mock
def test_api_key_env_fallback(monkeypatch):
    monkeypatch.setenv("ENGRAM_API_KEY", API_KEY)
    monkeypatch.setenv("ENGRAM_BASE_URL", BASE_URL)

    respx.get(f"{BASE_URL}/v1/memories/recall").mock(
        return_value=httpx.Response(200, json=_RECALL_EMPTY)
    )

    # Neither api_key nor base_url passed — must read from env
    memory = EngramChatMemory(agent_id=AGENT_ID)
    result = memory.load_memory_variables({"input": "test"})
    assert result["history"] == ""


@respx.mock
def test_bearer_token_sent_in_request():
    route = respx.get(f"{BASE_URL}/v1/memories/recall").mock(
        return_value=httpx.Response(200, json=_RECALL_EMPTY)
    )

    memory = EngramChatMemory(agent_id=AGENT_ID, api_key=API_KEY, base_url=BASE_URL)
    memory.load_memory_variables({"input": "test"})

    assert route.called
    auth_header = route.calls[0].request.headers.get("authorization", "")
    assert auth_header == f"Bearer {API_KEY}"


# ---------------------------------------------------------------------------
# memory_variables property
# ---------------------------------------------------------------------------


def test_memory_variables_property():
    memory = EngramChatMemory(agent_id=AGENT_ID, api_key=API_KEY, base_url=BASE_URL)
    assert memory.memory_variables == ["history"]


def test_custom_memory_key_property():
    memory = EngramChatMemory(
        agent_id=AGENT_ID, api_key=API_KEY, base_url=BASE_URL, memory_key="ctx"
    )
    assert memory.memory_variables == ["ctx"]
