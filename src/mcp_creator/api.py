from __future__ import annotations

from typing import Any

import uvicorn
from fastapi import FastAPI, Query
from pydantic import BaseModel, Field

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
async def spec_search(
    query: str = Query(...), k: int = Query(6, ge=1, le=20)
) -> list[SearchResult]:
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
        "instructions": "Answer strictly from the provided context; cite (title,/id,URL).",
    }


@app.post("/server/scaffold")
def server_scaffold(req: ScaffoldRequest) -> dict[str, Any]:
    tools_code = "".join(
        f"@mcp.tool()\n"
        f"def {t.name}({', '.join([f'{k}: {v}' for k,v in t.params.items()])}) -> str:\n"
        f'    """{t.docstring}"""\n'
        f"    return 'TODO: implement {t.name}'\n\n"
        for t in req.tools
    )
    res_code = "".join(
        f"@mcp.resource('spec://{r}')\n" f"def res_{i}() -> str:\n" f"    return 'resource {r}'\n\n"
        for i, r in enumerate(req.resources)
    )
    prm_code = "".join(
        f"@mcp.prompt()\n" f"def prompt_{i}() -> str:\n" f"    return 'Use this prompt: {p}'\n\n"
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
    pyproject_content = (
        "[project]\n"
        'name = "generated-mcp-server"\n'
        'version = "0.0.1"\n'
        'requires-python = ">=3.10"\n'
        'dependencies = ["mcp[cli]>=1.2.0"]\n'
    )
    files = [
        {"path": "server.py", "content": server_py},
        {"path": "pyproject.toml", "content": pyproject_content},
    ]
    return {"files": files, "notes": "Edit tool bodies and add tests."}


def main():  # pragma: no cover
    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
