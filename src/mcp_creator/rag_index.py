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
