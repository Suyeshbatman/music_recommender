from pathlib import Path

from src.retriever import Retriever, default_retriever


def test_default_retriever_loads_corpus():
    r = default_retriever()
    assert len(r.docs) > 0
    sources = {d.source for d in r.docs}
    assert "genres" in sources
    assert "artists" in sources
    assert "moods" in sources


def test_search_returns_relevant_genre_doc():
    r = default_retriever()
    results = r.search("lofi study beats for focus", k=3)
    assert len(results) > 0
    top_doc_id = results[0][0]
    # Top hit should be a lofi-related doc (either genres:lofi or an artist)
    combined = " ".join(text for _, _, text in results).lower()
    assert "lofi" in combined


def test_search_empty_query_returns_empty():
    r = default_retriever()
    assert r.search("", k=3) == []


def test_search_unknown_query_returns_empty_or_low():
    r = default_retriever()
    # A query with no catalog-relevant terms should return nothing
    results = r.search("xyzzyplugh", k=3)
    assert results == []


def test_retriever_on_empty_dir(tmp_path: Path):
    empty = tmp_path / "empty_kb"
    empty.mkdir()
    r = Retriever(empty)
    assert r.docs == []
    assert r.search("anything", k=3) == []


def test_format_context_handles_empty():
    r = default_retriever()
    assert "no relevant" in r.format_context([]).lower()


def test_format_context_renders_results():
    r = default_retriever()
    results = r.search("edm festival energy", k=2)
    text = r.format_context(results)
    assert "relevance" in text
    assert len(text) > 50
