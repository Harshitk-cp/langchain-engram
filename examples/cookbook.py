"""
langchain-engram cookbook
=========================

This cookbook shows how to use Engram's three LangChain integration classes:

    EngramChatMemory   — persists raw conversation turns, recalls semantically
    EngramEntityMemory — extracts structured entities/facts via LLM, recalls semantically
    EngramRetriever    — retrieves memories as LangChain Documents for RAG

Prerequisites
-------------
1. A running Engram server (self-hosted or Engram Cloud):
       docker run -p 8080:8080 engramai/engram

2. Environment variables:
       export ENGRAM_BASE_URL=http://localhost:8080
       export ENGRAM_API_KEY=mk_your_master_key
       export OPENAI_API_KEY=sk_...

3. An agent created in Engram:
       from engram import Engram
       client = Engram()
       agent = client.agents.create(external_id="demo-bot", name="Demo Bot")
       AGENT_ID = agent.id

Install
-------
    pip install langchain-engram langchain langchain-openai
"""

import os

# ---------------------------------------------------------------------------
# Shared setup
# ---------------------------------------------------------------------------

AGENT_ID = os.environ.get("ENGRAM_AGENT_ID", "replace-with-your-agent-uuid")

# api_key and base_url are read from ENGRAM_API_KEY / ENGRAM_BASE_URL env vars
# when not passed explicitly to each class.


# ===========================================================================
# Part 1: EngramChatMemory
# ===========================================================================
#
# Drop-in replacement for ConversationBufferMemory.
#
# What it does:
#   save_context  → stores each human/AI turn directly as Engram memories
#   load_memory   → recalls semantically relevant memories for the current input
#
# Unlike a buffer, it doesn't grow unbounded — old/weak memories decay over
# time according to Engram's confidence model. Hot memories (recently used,
# high confidence) surface first.
#
# When to use: general-purpose conversation memory, chatbots, assistants.

from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI

from langchain_engram import EngramChatMemory

chat_memory = EngramChatMemory(
    agent_id=AGENT_ID,
    memory_key="history",       # injected into chain prompt as {history}
    return_messages=False,      # True → List[BaseMessage], False → formatted string
    top_k=8,
    min_confidence=0.5,         # filter out uncertain/decayed memories
)

chat_chain = ConversationChain(llm=ChatOpenAI(model="gpt-4o-mini"), memory=chat_memory)

# First session
# chat_chain.predict(input="I always prefer dark mode in all my tools")
# chat_chain.predict(input="I'm a backend engineer — Go and Rust mostly")

# Second session (new process, same agent_id) — memories are loaded from Engram
# response = chat_chain.predict(input="What do you know about my setup?")
# → "You prefer dark mode and work with Go and Rust as a backend engineer."


# ===========================================================================
# Part 2: EngramEntityMemory
# ===========================================================================
#
# Structured entity extraction using Engram's LLM extraction pipeline.
#
# What it does:
#   save_context  → sends the conversation turn to Engram's /extract endpoint,
#                   which uses an LLM to identify typed memories:
#                     "I prefer dark mode" → preference, 0.95 confidence
#                     "I'm a senior Go engineer" → fact, 0.92 confidence
#                     "We can't use Python 2" → constraint, 0.98 confidence
#                   Stores them automatically. Contradictions trigger Engram's
#                   belief-update logic (no extra code needed).
#   load_memory   → recalls structured entity facts relevant to the current input
#
# When to use: when you want structured entity tracking rather than raw dialogue.
#              Better for task-oriented agents that need clean, typed knowledge.

from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate

from langchain_engram import EngramEntityMemory

entity_memory = EngramEntityMemory(
    agent_id=AGENT_ID,
    memory_key="entities",      # injected into prompt as {entities}
    top_k=10,
    min_confidence=0.6,
)

entity_prompt = PromptTemplate(
    input_variables=["entities", "input"],
    template=(
        "You are a helpful engineering assistant.\n\n"
        "What you know about this user:\n{entities}\n\n"
        "Human: {input}\n"
        "AI:"
    ),
)

entity_chain = LLMChain(
    llm=ChatOpenAI(model="gpt-4o-mini"),
    prompt=entity_prompt,
    memory=entity_memory,
)

# entity_chain.predict(input="I work in Go and I need a REST API for a todo app")
# entity_chain.predict(input="What stack should I use for the database layer?")
# → LLM sees: "Known entities and facts:
#              - User is building a REST API todo app [fact, 93% confident]
#              - User works in Go [fact, 91% confident]"


# ===========================================================================
# Part 3: EngramRetriever
# ===========================================================================
#
# Use Engram as a LangChain retriever in any chain or agent.
#
# What it does:
#   _get_relevant_documents → calls Engram's hybrid vector + graph recall,
#                             returns memories as LangChain Documents with
#                             full metadata: confidence, tier, type, scores.
#
# When to use:
#   - RetrievalQA chains where you want memory-backed answers
#   - Agents that need to check what they know before answering
#   - Any place a LangChain BaseRetriever is accepted
#
# Document metadata includes:
#   memory_id, agent_id, memory_type, confidence, tier (hot/warm/cold),
#   source, score, vector_score, graph_score, created_at

from langchain.chains import RetrievalQA

from langchain_engram import EngramRetriever

retriever = EngramRetriever(
    agent_id=AGENT_ID,
    top_k=5,
    min_confidence=0.6,         # only surface confident memories
    memory_type=None,           # "fact" | "preference" | "decision" | "constraint" | None
    graph_weight=None,          # 0–1; higher → weight graph traversal more
)

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4o-mini"),
    retriever=retriever,
)

# answer = qa_chain.invoke({"query": "What are the user's display preferences?"})

# Standalone retrieval — useful for inspecting what the agent knows:
# docs = retriever.invoke("Go programming preferences")
# for doc in docs:
#     print(f"{doc.page_content}")
#     print(f"  type: {doc.metadata['memory_type']}")
#     print(f"  confidence: {doc.metadata['confidence']:.0%}")
#     print(f"  tier: {doc.metadata['tier']}")
#     print(f"  score: {doc.metadata.get('score', 'n/a'):.3f}")


# ===========================================================================
# Part 4: Choosing between the three classes
# ===========================================================================
#
# EngramChatMemory
#   Use when: you want a direct conversation-history replacement, chatbots,
#             assistants where the raw dialogue is what matters.
#   Storage:  raw human/AI turns stored as memories, recalled semantically.
#   Tradeoff: stores more data (full turns), no entity structuring.
#
# EngramEntityMemory
#   Use when: you want structured entity/fact tracking, task-oriented agents,
#             or you want Engram's LLM to decide what's worth remembering.
#   Storage:  LLM-extracted typed memories (preference/fact/decision/constraint).
#   Tradeoff: one extra LLM call per turn (the extraction), but cleaner context.
#
# EngramRetriever
#   Use when: you need memory-backed RAG, or you want to feed memories into
#             an existing retrieval chain without the full memory interface.
#   Storage:  read-only — retrieves, does not save.
#   Tradeoff: no save_context; combine with a memory class if you need both.
#
# Combining EngramEntityMemory + EngramRetriever:
#   Use entity_memory in your chain (saves structured facts), and separately
#   use the retriever in tool-use or sub-chain contexts to query memories
#   without triggering another save_context cycle.


# ===========================================================================
# Part 5: API key scopes
# ===========================================================================
#
# Engram uses prefixed API keys:
#
#   mk_<64 hex>  master key — admin + read + write (created at tenant setup)
#   rk_<64 hex>  restricted key — user-chosen scopes
#
# For LangChain usage, a restricted key with ["read", "write"] scopes is all
# you need. Create one with:
#
#   from engram import Engram
#   client = Engram()  # uses mk_ master key from env
#   result = client.keys.create(name="langchain-bot", scopes=["read", "write"])
#   print(result.api_key)  # store this — shown only once
#
# Then set ENGRAM_API_KEY=rk_<that key> in your LangChain app environment.
