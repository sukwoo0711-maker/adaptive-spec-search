# Adaptive Spec Search

Local-first reference implementation for searching technical specifications
with lexical retrieval, optional dense embeddings, rank fusion, conservative
query expansion, metadata filters, and explicit relevance feedback.

This repository is an architecture example, not a model leaderboard. It does
not include private documents, private queries, or unpublished evaluation
results.

## Why multiple retrieval paths

Technical document queries often mix exact identifiers with natural language.
Keyword retrieval is useful for product codes, signal names, units, and exact
phrases. Dense retrieval is useful when the wording differs. Reciprocal Rank
Fusion combines their ranks without assuming their raw scores are calibrated.

```text
query
  -> normalization and audited synonym expansion
  -> BM25 candidate retrieval
  -> optional dense candidate retrieval
  -> Reciprocal Rank Fusion
  -> metadata and access-control checks
  -> optional cross-encoder reranking
  -> evidence with source identifiers
```

Query expansion is intentionally dictionary-based and capped. Unconstrained
expansion can reduce precision, particularly for numeric, identifier-heavy, or
ambiguous queries.

## Feedback without immediate overfitting

The included feedback mechanism applies a small query-overlap boost for an
explicitly marked relevant document and a penalty for a marked non-relevant
document. Production systems should keep three layers separate:

1. Session feedback for immediate re-search.
2. User or team preferences with bounded scope and expiry.
3. Globally promoted training examples accepted only after review, minimum
   support, offline replay, and holdout evaluation.

Clicks alone should not be treated as relevance labels. Prefer explicit
`relevant`, `not relevant`, `wrong version`, `wrong product`, and `correct
document` events. Non-relevant results that share many query terms are useful
hard negatives for a later learning-to-rank or reranker training stage.

## Minimal example

```python
from adaptive_spec_search import Document, FeedbackEvent, HybridSearchEngine

documents = [
    Document("REQ-1", "Scheduled power-off after the configured timer expires"),
    Document("REQ-2", "Room reservation and booking policy"),
]

engine = HybridSearchEngine(
    documents,
    synonyms={"shutdown": ["power off"], "예약": ["scheduled", "reservation"]},
)

results = engine.search("예약 후 자동 shutdown")
engine.add_feedback(FeedbackEvent("예약 후 자동 shutdown", "REQ-1", True))
```

An embedding function can be injected without coupling the package to a model
provider. It receives a list of strings and returns one numeric vector per
string.

## Evaluation protocol

Do not tune and report on the same questions. Freeze a test set before changing
weights, synonyms, chunking, or models.

- Split by document family, product, or intent, not random paraphrase.
- Keep exact-name, alias, description-only, numeric, multi-hop, contradictory,
  and unanswerable strata.
- Report Recall@k, MRR, nDCG, abstention, and worst-group performance.
- Compare variants on paired cases and use cluster-aware uncertainty intervals.
- Measure lexical, dense, fusion, reranking, and final-answer stages separately.
- Replay accepted feedback against a shadow index before promoting it globally.
- Verify access control before retrieval output reaches reranking or generation.

## Neutral deployment guidance

Hybrid retrieval, reranking, and query expansion solve different failure modes.
Adding every component does not guarantee improvement. Start from measured
errors: add lexical retrieval when exact terms are missed, add a reranker when
relevant candidates are present but poorly ordered, and add query expansion
only for demonstrated vocabulary gaps.

Local, hosted, and hybrid deployments should be compared on the same workload,
including quality, latency, throughput, memory, energy, security, maintenance,
and failure recovery.

## Development

```powershell
python -m pip install -e ".[dev]"
python -m pytest -q
```

## Status

Experimental reference implementation. Review the scoring, persistence,
authorization, and evaluation design before production use.

An empty result means “no eligible candidate was retrieved by this configured
pipeline,” not “the specification has no answer.” Scores are ranking signals,
not probabilities. The package itself makes no network request, but an injected
embedding callable can transmit documents and queries; confidential deployments
should use a trusted local implementation or an explicitly approved provider.

See [architecture and evaluation notes](docs/ARCHITECTURE.md) for feedback
promotion, overfitting controls, security boundaries, and staged evaluation.
See [the adversarial review](AUTO_GRILL.md) and
[deployment decisions](PORTING-DECISION-REQUIRED.md) before reuse.
