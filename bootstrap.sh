#!/usr/bin/env bash
set -euo pipefail

ROOT="$(pwd)"
echo "Scaffolding in: $ROOT"

# --- layout
mkdir -p src/mcp_creator tests scripts .github/workflows data

# --- .gitignore
cat > .gitignore <<'GI'
# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.venv/
.env
*.egg-info/
.build/
dist/
build/

# Tools & caches
.pytest_cache/
.ruff_cache/
.mypy_cache/
.cache/

# OS/editor
.DS_Store
.idea/
.vscode/

# Data
data/
GI

# --- LICENSE (MIT)
cat > LICENSE <<'LIC'
MIT License

Copyright (c) 2025

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
LIC

# --- README
cat > README.md <<'MD'
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
MD

# --- pyproject.toml

cat > pyproject.toml <<'TOML'
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "mcp-creator"
version = "0.1.0"
description = "RAG-backed MCP builder (API + optional MCP stdio server)"
requires-python = ">=3.10"
readme = "README.md"
license = { text = "MIT" }
authors = [{ name = "human-ai-colab" }]

dependencies = [
    "fastapi>=0.111.0",
    "uvicorn>=0.30.0",
    "httpx>=0.27.0",
    "pydantic>=2.7.0",
    "scikit-learn>=1.4.0"
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
    "ruff>=0.5.0",
    "black>=24.4.0",
    "requests>=2.32.0"
]
mcp = [
    "mcp[cli]>=1.2.0"
]

[project.scripts]
mcp-creator-api = "mcp_creator.api:main"
mcp-builder-server = "mcp_creator.mcp_builder_server:main"

[tool.setuptools]
package-dir = {"" = "src"}

[tool.setuptools.packages.find]
where = ["src"]

[tool.ruff]
line-length = 100
target-version = "py310"
select = ["E","F","I","UP","B"]

[tool.black]
line-length = 100
target-version = ["py310"]

[tool.pytest.ini_options]
addopts = "-q"
testpaths = ["tests"]
TOML

# --- src/mcp_creator/__init__.py

cat > src/mcp_creator/__init__.py <<'PY'
__all__ = ["rag_index"]
__version__ = "0.1.0"
PY

# --- src/mcp_creator/rag_index.py (TF‑IDF baseline; no heavy dependencies)

cat > src/mcp_creator/rag_index.py <<'PY'
from __future__ import annotations
import os, re, hashlib, pathlib, asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

import httpx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

LLMS_FULL_URL = "https://modelcontextprotocol.io/llms-full.txt"
CACHE_DIR = os.environ.get("MCP_CREATOR_CACHE", "data")

@dataclass
class Chunk:
    id: str
    title: str
    url: str
    text: str
    parent_titles: list[str]
    order: int
    source: str = "llms-full.txt"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

def _hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]

def split_markdown_sections(text: str) -> list[tuple[list[str], str]]:
    """Return [(heading_hierarchy, section_text), …] from markdown."""
    lines = text.splitlines()
    sections: list[tuple[list[str], str]] = []
    current_h: list[str] = []
    buf: list[str] = []
    hdr_re = re.compile(r"^(#{1,6})\s+(.*)")
    for ln in lines:
        m = hdr_re.match(ln)
        if m:
            if buf:
                sections.append((current_h.copy(), "\n".join(buf).strip()))
                buf = []
            level = len(m.group(1))
            title = m.group(2).strip()
            current_h = current_h[: level - 1] + [title]
        else:
            buf.append(ln)
    if buf:
        sections.append((current_h.copy(), "\n".join(buf).strip()))
    return sections

async def download_llms_full(cache_dir: str = CACHE_DIR) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    path = pathlib.Path(cache_dir) / "llms-full.txt"
    if path.exists() and path.stat().st_size > 0:
        return str(path)
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.get(LLMS_FULL_URL)
        r.raise_for_status()
        path.write_text(r.text, encoding="utf-8")
    return str(path)

def build_chunks(llms_text: str, window_chars: int = 5000) -> list[Chunk]:
    sections = split_markdown_sections(llms_text)
    chunks: list[Chunk] = []
    order = 0
    for heads, body in sections:
        if not body.strip():
            continue
        for i in range(0, len(body), window_chars):
            piece = body[i : i + window_chars]
            title = heads[-1] if heads else "Untitled"
            cid = _hash("|".join(heads) + f"#{i}")
            chunks.append(
                Chunk(
                    id=cid,
                    title=title,
                    url=LLMS_FULL_URL,
                    text=piece.strip(),
                    parent_titles=heads[:-1],
                    order=order,
                )
            )
            order += 1
    return chunks

class RAGIndex:
    """Simple TF‑IDF index over Chunks."""
    def __init__(self):
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.doc_matrix = None
        self.docs: list[str] = []
        self.meta: list[Dict[str, Any]] = []

    def fit(self, chunks: list[Chunk]) -> "RAGIndex":
        self.docs = [c.text for c in chunks]
        self.meta = [c.to_dict() for c in chunks]
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.doc_matrix = self.vectorizer.fit_transform(self.docs)
        return self

    def query(self, query: str, k: int = 6) -> list[Dict[str, Any]]:
        if not self.vectorizer or self.doc_matrix is None:
            raise RuntimeError("Index not built. Call fit() first.")
        q_vec = self.vectorizer.transform([query])
        sims = linear_kernel(q_vec, self.doc_matrix).ravel()
        top_idx = sims.argsort()[::-1][:k]
        out: list[Dict[str, Any]] = []
        for i in top_idx:
            item = dict(self.meta[i])
            item["score"] = float(sims[i])
            out.append(item)
        return out

async def bootstrap_index() -> RAGIndex:
    path = await download_llms_full()
    text = pathlib.Path(path).read_text(encoding="utf-8")
    chunks = build_chunks(text)
    idx = RAGIndex().fit(chunks)
    return idx
PY

# --- src/mcp_creator/mcp_builder_server.py (MCP stdio server - optional)

cat > src/mcp_creator/mcp_builder_server.py <<'PY'
from __future__ import annotations
import json
from typing import Any

try:
    from mcp.server.fastmcp import FastMCP  # type: ignore
except Exception as e:  # pragma: no cover
    FastMCP = None
    _IMPORT_ERR = e

from pydantic import BaseModel, Field
from .rag_index import bootstrap_index

class SearchResult(BaseModel):
    id: str
    title: str
    url: str
    text: str | None = None
    parent_titles: list[str]
    order: int
    score: float | None = None

class ScaffoldTool(BaseModel):
    name: str
    docstring: str
    params: dict[str, str] = Field(default_factory=dict)

class ScaffoldRequest(BaseModel):
    server_name: str
    tools: list[ScaffoldTool] = Field(default_factory=list)
    resources: list[str] = Field(default_factory=list)
    prompts: list[str] = Field(default_factory=list)

def _server_code(req: ScaffoldRequest) -> str:
    tools_code = "".join(
        f"@mcp.tool()\n"
        f"def {t.name}({', '.join([f'{k}: {v}' for k,v in t.params.items()])}) -> str:\n"
        f"    \"\"\"{t.docstring}\"\"\"\n"
        f"    return 'TODO: implement {t.name}'\n\n"
        for t in req.tools
    )
    res_code = "".join(
        f"@mcp.resource('spec://{r}')\n"
        f"def res_{i}() -> str:\n"
        f"    return 'resource {r}'\n\n"
        for i, r in enumerate(req.resources)
    )
    prm_code = "".join(
        f"@mcp.prompt()\n"
        f"def prompt_{i}() -> str:\n"
        f"    return 'Use this prompt: {p}'\n\n"
        for i, p in enumerate(req.prompts)
    )
    return (
        "from mcp.server.fastmcp import FastMCP\n\n"
        f"mcp = FastMCP('{req.server_name}')\n\n"
        f"{tools_code}{res_code}{prm_code}"
        "def main():\n"
        "    mcp.run(transport='stdio')\n\n"
        "if __name__ == '__main__':\n"
        "    main()\n"
    )

def main():  # pragma: no cover
    if FastMCP is None:
        print("""
This is an MCP stdio server stub.
To enable it, install the MCP Python package:

    pip install "mcp[cli]>=1.2.0"

Then run:
    python -m mcp_creator.mcp_builder_server
""")
        print(f"Import error: {_IMPORT_ERR}")
        return

    mcp = FastMCP("mcp-creator")

    @mcp.tool()
    async def spec_search(query: str, k: int = 6) -> list[SearchResult]:
        idx = await bootstrap_index()
        rows = idx.query(query, k=k)
        # don't stuff full text into the wire unless needed
        return [SearchResult(**r) for r in rows]

    @mcp.tool()
    async def spec_answer(question: str) -> dict[str, Any]:
        idx = await bootstrap_index()
        rows = idx.query(question, k=8)
        citations = [
            {"title": r["title"], "parents": r["parent_titles"], "url": r["url"], "id": r["id"]}
            for r in rows
        ]
        context = "\\n\\n---\\n\\n".join(
            f"# {r['title']} (…/{r['id']})\\n{r['text'][:2000]}\\n\\nSource: {r['url']}"
            for r in rows
        )
        return {
            "context": context,
            "citations": citations,
            "instructions": (
                "Answer strictly from the provided context and cite (title,/id,URL)."
            ),
        }

    @mcp.tool()
    def server_scaffold(server_name: str, tools_json: str = "[]", resources_json: str = "[]", prompts_json: str = "[]") -> dict[str, Any]:
        tools = [ScaffoldTool(**t) for t in json.loads(tools_json)]
        req = ScaffoldRequest(
            server_name=server_name,
            tools=tools,
            resources=json.loads(resources_json),
            prompts=json.loads(prompts_json),
        )
        files = [
            {"path": "server.py", "content": _server_code(req)},
            {"path": "pyproject.toml", "content": '[project]\\nname = "generated-mcp-server"\\nversion = "0.0.1"\\nrequires-python = ">=3.10"\\ndependencies = ["mcp[cli]>=1.2.0"]\\n'}
        ]
        return {"files": files, "notes": "Edit tool bodies and add tests."}

    mcp.run(transport="stdio")

if __name__ == "__main__":
    main()
PY

# --- src/mcp_creator/api.py (FastAPI)

cat > src/mcp_creator/api.py <<'PY'
from __future__ import annotations
from typing import Any
from pydantic import BaseModel, Field
from fastapi import FastAPI, Query
import uvicorn

from .rag_index import bootstrap_index

app = FastAPI(title="mcp-creator API", version="0.1.0")

class SearchResult(BaseModel):
    id: str
    title: str
    url: str
    parent_titles: list[str]
    order: int
    score: float | None = None

class ScaffoldTool(BaseModel):
    name: str
    docstring: str
    params: dict[str, str] = Field(default_factory=dict)

class ScaffoldRequest(BaseModel):
    server_name: str
    tools: list[ScaffoldTool] = Field(default_factory=list)
    resources: list[str] = Field(default_factory=list)
    prompts: list[str] = Field(default_factory=list)

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

@app.get("/spec/search")
async def spec_search(query: str = Query(...), k: int = Query(6, ge=1, le=20)) -> list[SearchResult]:
    idx = await bootstrap_index()
    rows = idx.query(query, k=k)
    return [SearchResult(**r) for r in rows]

@app.post("/spec/answer")
async def spec_answer(payload: dict[str, Any]) -> dict[str, Any]:
    question = payload.get("question", "")
    if not question:
        return {"error": "question is required"}
    idx = await bootstrap_index()
    rows = idx.query(question, k=8)
    citations = [
        {"title": r["title"], "parents": r["parent_titles"], "url": r["url"], "id": r["id"]}
        for r in rows
    ]
    context = "\n\n---\n\n".join(
        f"# {r['title']} (…/{r['id']})\n{r['text']}\n\nSource: {r['url']}" for r in rows
    )
    return {
        "context": context,
        "citations": citations,
        "instructions": "Answer strictly from the provided context; cite (title,/id,URL)."
    }

@app.post("/server/scaffold")
def server_scaffold(req: ScaffoldRequest) -> dict[str, Any]:
    tools_code = "".join(
        f"@mcp.tool()\n"
        f"def {t.name}({', '.join([f'{k}: {v}' for k,v in t.params.items()])}) -> str:\n"
        f"    \"\"\"{t.docstring}\"\"\"\n"
        f"    return 'TODO: implement {t.name}'\n\n"
        for t in req.tools
    )
    res_code = "".join(
        f"@mcp.resource('spec://{r}')\n"
        f"def res_{i}() -> str:\n"
        f"    return 'resource {r}'\n\n"
        for i, r in enumerate(req.resources)
    )
    prm_code = "".join(
        f"@mcp.prompt()\n"
        f"def prompt_{i}() -> str:\n"
        f"    return 'Use this prompt: {p}'\n\n"
        for i, p in enumerate(req.prompts)
    )
    server_py = (
        "from mcp.server.fastmcp import FastMCP\n\n"
        f"mcp = FastMCP('{req.server_name}')\n\n"
        f"{tools_code}{res_code}{prm_code}"
        "def main():\n"
        "    mcp.run(transport='stdio')\n\n"
        "if __name__ == '__main__':\n"
        "    main()\n"
    )
    files = [
        {"path": "server.py", "content": server_py},
        {"path": "pyproject.toml", "content": '[project]\nname = "generated-mcp-server"\nversion = "0.0.1"\nrequires-python = ">=3.10"\ndependencies = ["mcp[cli]>=1.2.0"]\n'}
    ]
    return {"files": files, "notes": "Edit tool bodies and add tests."}

def main():  # pragma: no cover
    uvicorn.run(app, host="127.0.0.1", port=8000)

if __name__ == "__main__":
    main()
PY

# --- tests

cat > tests/__init__.py <<'PY'
PY

cat > tests/test_chunker.py <<'PY'
from mcp_creator.rag_index import split_markdown_sections, build_chunks, RAGIndex

def test_split_and_query():
    md = "# A\ntext a\n\n## A1\ntext a1\n\n# B\ntext b"
    secs = split_markdown_sections(md)
    assert len(secs) >= 2
    chunks = build_chunks(md, window_chars=1000)
    idx = RAGIndex().fit(chunks)
    res = idx.query("text b", k=1)
    assert res and res[0]["title"] in {"B", "A1", "A"}
PY

# --- CI workflow

cat > .github/workflows/ci.yml <<'YML'
name: CI
on:
  push:
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install
        run: |
          python -m venv .venv
          source .venv/bin/activate
          python -m pip install -U pip
          pip install -e ".[dev]"
      - name: Lint
        run: |
          source .venv/bin/activate
          ruff check .
          black --check .
      - name: Test
        run: |
          source .venv/bin/activate
          pytest -q
YML

# --- simple run helpers

cat > scripts/dev.sh <<'SH'
#!/usr/bin/env bash
set -euo pipefail
python -m mcp_creator.api
SH
chmod +x scripts/dev.sh

# --- git init & first commit

if [ ! -d ".git" ]; then
    git init
fi
git add -A
git commit -m "chore: scaffold mcp-creator (RAG + API + MCP stubs)"

# --- create GitHub repo (optional; requires gh)

if command -v gh >/dev/null 2>&1; then
    if ! git remote get-url origin >/dev/null 2>&1; then
        gh repo create mcp-creator --public --source=. --remote=origin --push
    else
        echo "Remote 'origin' already set, skipping gh repo create."
    fi
else
    echo "gh CLI not found; skipping GitHub remote creation. You can add it later:"
    echo "  gh repo create mcp-creator --public --source=. --remote=origin --push"
fi

echo ""
echo "Bootstrap complete."
echo "Next:"
echo "  1) source .venv/bin/activate && pip install -e '.[dev]'"
echo "  2) ./scripts/dev.sh   # open http://127.0.0.1:8000/docs"
echo "  3) (optional) pip install 'mcp[cli]>=1.2.0' && python -m mcp_creator.mcp_builder_server"
