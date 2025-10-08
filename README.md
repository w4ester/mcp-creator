# mcp-creator

RAG-backed MCP builder: turn MCP spec/context into grounded answers and scaffolds for **MCP servers** (tools/resources/prompts), with a simple HTTP API. Optional MCP stdio server included.

## Quick Start

```bash
# create venv + install
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"

# run API (http://127.0.0.1:8000/docs)
python -m mcp_creator.api
```

## Endpoints
    •    GET /health
    •    GET /spec/search?query=...&k=6
    •    POST /spec/answer → { "question": "..." }
    •    POST /server/scaffold → { server_name, tools[], resources[], prompts[] }

## Optional: MCP server

Install extra and run:

```bash
pip install "mcp[cli]>=1.2.0" || true  # if available on PyPI
python -m mcp_creator.mcp_builder_server  # prints instructions if MCP not installed
```

## Tests

```bash
pytest -q
```

## Notes
    •    RAG: TF‑IDF baseline (no GPU). Easily swap to dense vectors/chroma later.
    •    The server fetches the official MCP long-form text llms-full.txt at first use and caches it.
