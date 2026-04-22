"""Pure-Python TF-IDF retriever over the data/knowledge markdown corpus.

No external deps. Paragraph-level chunks keyed by the nearest `##` heading.
"""

import math
import re
from pathlib import Path
from typing import Dict, List, Tuple

_WORD_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "for", "with",
    "is", "are", "was", "were", "be", "been", "being", "as", "at", "by", "from",
    "this", "that", "these", "those", "it", "its", "their", "some", "any",
    "very", "often", "usually", "than", "also", "here", "here,",
}


def _tokenize(text: str) -> List[str]:
    return [w for w in _WORD_RE.findall(text.lower()) if w not in _STOPWORDS]


class Document:
    __slots__ = ("doc_id", "source", "heading", "text", "tf")

    def __init__(self, doc_id: str, source: str, heading: str, text: str):
        self.doc_id = doc_id
        self.source = source
        self.heading = heading
        self.text = text
        self.tf: Dict[str, int] = {}
        for tok in _tokenize(f"{heading} {text}"):
            self.tf[tok] = self.tf.get(tok, 0) + 1


class Retriever:
    """Loads the knowledge corpus on construction and serves top-k queries."""

    def __init__(self, knowledge_dir: Path):
        self.knowledge_dir = Path(knowledge_dir)
        self.docs: List[Document] = []
        self.idf: Dict[str, float] = {}
        self._load()

    def _load(self) -> None:
        if not self.knowledge_dir.exists():
            return
        for md_file in sorted(self.knowledge_dir.glob("*.md")):
            self._parse_markdown(md_file)
        self._compute_idf()

    def _parse_markdown(self, path: Path) -> None:
        """Split a markdown file on `## ` headings. Each section is a document."""
        source = path.stem
        content = path.read_text(encoding="utf-8")
        current_heading = ""
        current_body: List[str] = []

        def flush():
            if current_heading and current_body:
                body = "\n".join(current_body).strip()
                if body:
                    doc_id = f"{source}:{current_heading}"
                    self.docs.append(Document(doc_id, source, current_heading, body))

        for line in content.splitlines():
            if line.startswith("## "):
                flush()
                current_heading = line[3:].strip()
                current_body = []
            elif line.startswith("# "):
                continue
            else:
                current_body.append(line)
        flush()

    def _compute_idf(self) -> None:
        n = len(self.docs)
        if n == 0:
            return
        df: Dict[str, int] = {}
        for doc in self.docs:
            for term in doc.tf:
                df[term] = df.get(term, 0) + 1
        self.idf = {
            term: math.log((n + 1) / (freq + 1)) + 1.0
            for term, freq in df.items()
        }

    def _score(self, query_tokens: List[str], doc: Document) -> float:
        if not doc.tf:
            return 0.0
        score = 0.0
        doc_len = sum(doc.tf.values())
        for tok in query_tokens:
            if tok in doc.tf:
                tf = doc.tf[tok] / doc_len
                score += tf * self.idf.get(tok, 0.0)
        return score

    def search(self, query: str, k: int = 3) -> List[Tuple[str, float, str]]:
        """Return up to k (doc_id, score, text) tuples sorted by score desc."""
        if not self.docs:
            return []
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []
        scored = [
            (doc, self._score(query_tokens, doc))
            for doc in self.docs
        ]
        scored = [s for s in scored if s[1] > 0]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [
            (doc.doc_id, score, doc.text)
            for doc, score in scored[:k]
        ]

    def format_context(self, results: List[Tuple[str, float, str]]) -> str:
        """Render retrieval results as a compact context block for the LLM."""
        if not results:
            return "(no relevant knowledge base entries)"
        lines = []
        for doc_id, score, text in results:
            lines.append(f"### {doc_id} (relevance {score:.2f})")
            lines.append(text.strip())
            lines.append("")
        return "\n".join(lines).strip()


def default_retriever() -> Retriever:
    """Build a retriever pointed at the repo's data/knowledge directory."""
    repo_root = Path(__file__).resolve().parent.parent
    return Retriever(repo_root / "data" / "knowledge")
