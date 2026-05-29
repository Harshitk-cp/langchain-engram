# langchain-engram

LangChain integration for [Engram](https://engram.ai) — cognitive memory infrastructure for AI agents.

Engram stores memories with **confidence scores**, detects **contradictions**, and manages **memory lifecycle** (decay, consolidation, tier promotion). Unlike a conversation buffer, it surfaces what the agent is most confident about — not just what was said most recently.

## Installation

```bash
pip install langchain-engram
```

This also installs `engram.to`, the Engram Python SDK.

## Setup

You need a running Engram server and an API key. Keys are prefixed `mk_` (master, all scopes) or `rk_` (restricted, user-chosen scopes).

```bash
export ENGRAM_BASE_URL=http://localhost:8080   # your Engram server
export ENGRAM_API_KEY=mk_your_key_here
```

Or pass them explicitly to each class.

## EngramChatMemory

Drop-in replacement for `ConversationBufferMemory`. Stores each conversation turn as Engram memories and retrieves semantically relevant context on the next turn — not a flat buffer.

```python
from langchain_engram import EngramChatMemory
from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI

memory = EngramChatMemory(
    agent_id="your-agent-uuid",
    # api_key and base_url read from env if not passed
)

chain = ConversationChain(llm=ChatOpenAI(), memory=memory)

chain.predict(input="I always prefer dark mode in all my tools")
chain.predict(input="I'm a backend engineer, mostly Go and Rust")

# Next session — memories are recalled from Engram, not kept in RAM
response = chain.predict(input="What do you know about my preferences?")
```

### Return as messages

```python
memory = EngramChatMemory(
    agent_id="your-agent-uuid",
    return_messages=True,   # returns List[BaseMessage] instead of a string
    min_confidence=0.7,     # only confident memories
    top_k=5,
)
```

### Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `agent_id` | required | Engram agent UUID |
| `api_key` | `ENGRAM_API_KEY` env | `mk_` or `rk_` prefixed API key |
| `base_url` | `ENGRAM_BASE_URL` env | Engram server URL |
| `memory_key` | `"history"` | Key injected into chain inputs |
| `input_key` | auto-detected | Key in `inputs` holding the human message |
| `output_key` | auto-detected | Key in `outputs` holding the AI response |
| `return_messages` | `False` | Return `List[BaseMessage]` instead of a string |
| `top_k` | `10` | Max memories to retrieve per turn |
| `min_confidence` | `None` | Confidence floor — filters out uncertain memories |

## EngramRetriever

Use Engram as a retriever in any LangChain chain or agent. Retrieves memories using Engram's hybrid vector + knowledge-graph recall.

```python
from langchain_engram import EngramRetriever
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

retriever = EngramRetriever(
    agent_id="your-agent-uuid",
    top_k=5,
    min_confidence=0.6,
)

qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(), retriever=retriever)
answer = qa.invoke({"query": "What are the user's display preferences?"})
```

Each retrieved `Document` carries full metadata:

```python
docs = retriever.invoke("display preferences")
for doc in docs:
    print(doc.page_content)
    print(f"  confidence: {doc.metadata['confidence']:.0%}")
    print(f"  tier: {doc.metadata['tier']}")        # hot / warm / cold
    print(f"  type: {doc.metadata['memory_type']}")  # fact / preference / decision
    print(f"  score: {doc.metadata['score']:.3f}")   # combined recall score
```

### Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `agent_id` | required | Engram agent UUID |
| `api_key` | `ENGRAM_API_KEY` env | API key |
| `base_url` | `ENGRAM_BASE_URL` env | Engram server URL |
| `top_k` | `10` | Max memories to return |
| `min_confidence` | `None` | Confidence floor |
| `memory_type` | `None` | Filter: `"fact"`, `"preference"`, `"decision"`, `"constraint"` |
| `graph_weight` | `None` | Graph/vector blend (0–1). Server default: 0.4 graph / 0.6 vector |

## API key scopes

| Key prefix | Scopes | Use case |
|------------|--------|----------|
| `mk_...` | admin + read + write | Server setup, key management |
| `rk_...` | user-chosen | Production agents (read + write is sufficient) |

For LangChain usage, a `rk_` key with `read` and `write` scopes is all you need.

## Links

- [Engram documentation](https://docs.engram.ai)
- [Python SDK (engram.to)](https://pypi.org/project/engram.to/)
- [GitHub](https://github.com/engram-ai/langchain-engram)
