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
