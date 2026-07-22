from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_DIR = BASE_DIR / "knowledge"


@dataclass
class RagChunk:
    source: str
    title: str
    text: str


class LocalRagEngine:
    def __init__(self, knowledge_dir: Path = KNOWLEDGE_DIR):
        self.knowledge_dir = knowledge_dir
        self._chunks: list[RagChunk] = []
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix = None
        self._loaded_signature: tuple[tuple[str, float], ...] = ()

    def _signature(self) -> tuple[tuple[str, float], ...]:
        if not self.knowledge_dir.exists():
            return ()
        return tuple(
            sorted(
                (str(path.relative_to(self.knowledge_dir)), path.stat().st_mtime)
                for path in self.knowledge_dir.glob("*.md")
            )
        )

    def _split_markdown(self, source: str, text: str) -> Iterable[RagChunk]:
        current_title = Path(source).stem
        buffer: list[str] = []

        def flush():
            body = "\n".join(buffer).strip()
            if body:
                yield RagChunk(source=source, title=current_title, text=body)

        for line in text.splitlines():
            if line.startswith("#"):
                yield from flush()
                buffer = []
                current_title = line.lstrip("#").strip() or current_title
            else:
                buffer.append(line)
        yield from flush()

    def _load(self):
        signature = self._signature()
        if signature == self._loaded_signature:
            return

        chunks: list[RagChunk] = []
        if self.knowledge_dir.exists():
            for path in sorted(self.knowledge_dir.glob("*.md")):
                raw = path.read_text(encoding="utf-8")
                chunks.extend(self._split_markdown(path.name, raw))

        self._chunks = chunks
        self._loaded_signature = signature
        if not chunks:
            self._vectorizer = None
            self._matrix = None
            return

        self._vectorizer = TfidfVectorizer(
            lowercase=True,
            analyzer="char_wb",
            ngram_range=(2, 5),
        )
        self._matrix = self._vectorizer.fit_transform(
            [f"{chunk.title}\n{chunk.text}" for chunk in chunks]
        )

    def search(self, query: str, top_k: int = 4) -> list[dict]:
        self._load()
        clean_query = (query or "").strip()
        if not clean_query or not self._chunks or self._vectorizer is None:
            return []

        q = self._vectorizer.transform([clean_query])
        scores = cosine_similarity(q, self._matrix).ravel()
        ranked = scores.argsort()[::-1][: max(1, int(top_k))]
        results = []
        for idx in ranked:
            score = float(scores[idx])
            if score <= 0:
                continue
            chunk = self._chunks[int(idx)]
            results.append(
                {
                    "source": chunk.source,
                    "title": chunk.title,
                    "score": round(score, 4),
                    "text": _compact(chunk.text),
                }
            )
        return results


def _compact(text: str, max_chars: int = 650) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 1].rstrip() + "…"


rag_engine = LocalRagEngine()


def retrieve_knowledge(query: str, top_k: int = 4) -> list[dict]:
    return rag_engine.search(query, top_k=top_k)
