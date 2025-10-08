from mcp_creator.rag_index import RAGIndex, build_chunks, split_markdown_sections


def test_split_and_query():
    md = "# A\ntext a\n\n## A1\ntext a1\n\n# B\ntext b"
    secs = split_markdown_sections(md)
    assert len(secs) >= 2
    chunks = build_chunks(md, window_chars=1000)
    idx = RAGIndex().fit(chunks)
    res = idx.query("text b", k=1)
    assert res and res[0]["title"] in {"B", "A1", "A"}
