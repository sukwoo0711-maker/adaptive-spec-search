"""Dependency-light hybrid retrieval primitives for private document search."""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping, Sequence

TOKEN_RE = re.compile(r"[가-힣]+|[A-Za-z]+(?:[._/-][A-Za-z0-9]+)*|\d+(?:\.\d+)?")


def tokenize(text: str) -> list[str]:
    """Tokenize Korean, identifiers, and numbers without external analyzers."""
    split_camel = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)
    return [match.group(0).casefold() for match in TOKEN_RE.finditer(split_camel)]


@dataclass(frozen=True)
class Document:
    id: str
    text: str
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchResult:
    document: Document
    score: float
    ranks: Mapping[str, int]


@dataclass(frozen=True)
class FeedbackEvent:
    query: str
    document_id: str
    relevant: bool
    reason: str = ""


class BM25Index:
    def __init__(self, documents: Sequence[Document], k1: float = 1.5, b: float = 0.75):
        self.documents = list(documents)
        self.k1, self.b = k1, b
        self.term_counts = [Counter(tokenize(doc.text)) for doc in documents]
        self.lengths = [sum(counts.values()) for counts in self.term_counts]
        self.average_length = sum(self.lengths) / max(1, len(self.lengths))
        document_frequency: Counter[str] = Counter()
        for counts in self.term_counts:
            document_frequency.update(counts.keys())
        total = len(documents)
        self.idf = {
            term: math.log(1 + (total - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def search(self, query: str, limit: int) -> list[tuple[str, float]]:
        query_terms = tokenize(query)
        scored = []
        for doc, counts, length in zip(self.documents, self.term_counts, self.lengths):
            score = 0.0
            for term in query_terms:
                frequency = counts.get(term, 0)
                if not frequency:
                    continue
                denominator = frequency + self.k1 * (
                    1 - self.b + self.b * length / max(1, self.average_length)
                )
                score += self.idf.get(term, 0.0) * frequency * (self.k1 + 1) / denominator
            if score:
                scored.append((doc.id, score))
        return sorted(scored, key=lambda item: (-item[1], item[0]))[:limit]


class HybridSearchEngine:
    """Fuse lexical and optional dense retrieval, then apply safe feedback boosts.

    ``embed`` must return one vector per input string. Keeping it injectable makes
    the core independent of a particular local runtime or embedding model.
    """

    def __init__(
        self,
        documents: Sequence[Document],
        *,
        embed: Callable[[Sequence[str]], Sequence[Sequence[float]]] | None = None,
        synonyms: Mapping[str, Sequence[str]] | None = None,
        rrf_k: int = 60,
        feedback_weight: float = 0.001,
        max_documents: int = 100_000,
        max_text_chars: int = 2_000_000,
    ):
        if not documents:
            raise ValueError("at least one document is required")
        if len(documents) > max_documents:
            raise ValueError("document limit exceeded")
        if len({doc.id for doc in documents}) != len(documents):
            raise ValueError("document ids must be unique")
        if any(not doc.id or len(doc.text) > max_text_chars for doc in documents):
            raise ValueError("document id is empty or document text limit exceeded")
        if rrf_k <= 0 or feedback_weight < 0:
            raise ValueError("rrf_k must be positive and feedback_weight non-negative")
        self.documents = {doc.id: doc for doc in documents}
        self.bm25 = BM25Index(documents)
        self.embed = embed
        self.rrf_k = rrf_k
        self.feedback_weight = feedback_weight
        self.synonyms = {key.casefold(): tuple(values) for key, values in (synonyms or {}).items()}
        self.feedback: list[FeedbackEvent] = []
        self.document_vectors = None
        if embed:
            self.document_vectors = list(embed([doc.text for doc in documents]))
            self._validate_vectors(self.document_vectors, len(documents))
            self._vector_ids = [doc.id for doc in documents]

    @staticmethod
    def _validate_vectors(vectors: Sequence[Sequence[float]], expected: int) -> None:
        if len(vectors) != expected or not vectors:
            raise ValueError("embedding function must return one vector per input")
        width = len(vectors[0])
        if width == 0:
            raise ValueError("embedding vectors must not be empty")
        if any(len(vector) != width for vector in vectors):
            raise ValueError("embedding vectors must have equal dimensions")
        if any(not math.isfinite(float(value)) for vector in vectors for value in vector):
            raise ValueError("embedding vectors must contain finite numeric values")

    def expand_query(self, query: str, max_terms: int = 8) -> str:
        """Apply an audited dictionary only; no unconstrained LLM expansion."""
        additions: list[str] = []
        for token in tokenize(query):
            for synonym in self.synonyms.get(token, ()):
                if synonym.casefold() not in query.casefold() and synonym not in additions:
                    additions.append(synonym)
                if len(additions) >= max_terms:
                    return query + " " + " ".join(additions)
        return query + (" " + " ".join(additions) if additions else "")

    def _dense_search(
        self, query: str, limit: int, eligible_ids: set[str]
    ) -> list[tuple[str, float]]:
        if not self.embed or self.document_vectors is None:
            return []
        query_rows = list(self.embed([query]))
        self._validate_vectors(query_rows, 1)
        query_vector = list(query_rows[0])
        if len(query_vector) != len(self.document_vectors[0]):
            raise ValueError("query and document embedding dimensions differ")
        query_norm = math.sqrt(sum(value * value for value in query_vector)) or 1.0
        scored = []
        for doc_id, vector in zip(self._vector_ids, self.document_vectors):
            if doc_id not in eligible_ids:
                continue
            norm = math.sqrt(sum(value * value for value in vector)) or 1.0
            score = sum(a * b for a, b in zip(query_vector, vector)) / (query_norm * norm)
            scored.append((doc_id, score))
        return sorted(scored, key=lambda item: (-item[1], item[0]))[:limit]

    def add_feedback(self, event: FeedbackEvent) -> None:
        if event.document_id not in self.documents:
            raise KeyError(event.document_id)
        self.feedback.append(event)

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        candidate_limit: int = 50,
        expand: bool = True,
        metadata_filter: Mapping[str, str] | None = None,
    ) -> list[SearchResult]:
        if not query.strip():
            return []
        if limit <= 0 or candidate_limit <= 0:
            raise ValueError("limit and candidate_limit must be positive")
        eligible_ids = {
            doc.id
            for doc in self.documents.values()
            if not metadata_filter
            or all(doc.metadata.get(key) == value for key, value in metadata_filter.items())
        }
        if not eligible_ids:
            return []
        retrieval_query = self.expand_query(query) if expand else query
        arms = {
            "bm25": [
                row
                for row in self.bm25.search(retrieval_query, len(self.documents))
                if row[0] in eligible_ids
            ][:candidate_limit],
            "dense": self._dense_search(query, candidate_limit, eligible_ids),
        }
        fused: defaultdict[str, float] = defaultdict(float)
        ranks: defaultdict[str, dict[str, int]] = defaultdict(dict)
        for arm, rows in arms.items():
            for rank, (doc_id, _) in enumerate(rows, 1):
                fused[doc_id] += 1 / (self.rrf_k + rank)
                ranks[doc_id][arm] = rank

        query_tokens = set(tokenize(query))
        feedback_adjustments: defaultdict[str, float] = defaultdict(float)
        for event in self.feedback:
            overlap = len(query_tokens & set(tokenize(event.query)))
            if overlap:
                direction = 1 if event.relevant else -1
                feedback_adjustments[event.document_id] += direction * self.feedback_weight * overlap
        for document_id, adjustment in feedback_adjustments.items():
            # Feedback refines an existing candidate. It never injects a document
            # that retrieval did not find, and one query cannot dominate RRF.
            if document_id in fused:
                fused[document_id] += max(-0.005, min(0.005, adjustment))

        results = []
        for doc_id, score in fused.items():
            document = self.documents[doc_id]
            results.append(SearchResult(document, score, ranks[doc_id]))
        return sorted(results, key=lambda row: (-row.score, row.document.id))[:limit]


def reciprocal_rank(results: Iterable[SearchResult], expected_id: str) -> float:
    for rank, result in enumerate(results, 1):
        if result.document.id == expected_id:
            return 1 / rank
    return 0.0
