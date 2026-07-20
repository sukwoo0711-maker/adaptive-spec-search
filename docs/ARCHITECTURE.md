# Architecture and evaluation notes

## Retrieval stages

The reference pipeline separates candidate recall from candidate precision.

1. Parse documents using their native hierarchy, tables, identifiers, and
   version metadata.
2. Normalize units, identifiers, and approved domain terminology.
3. Apply authorization and metadata constraints as early as the storage system
   permits.
4. Retrieve lexical and optional dense candidate lists independently.
5. Fuse ranks with Reciprocal Rank Fusion.
6. Optionally rerank a bounded candidate list with a query-document model.
7. Return source identifiers with evidence; generation is a separate stage.

Keyword, dense, query expansion, and reranking components address different
failure modes. They should remain independently observable so a degraded arm
cannot silently masquerade as hybrid search.

## Query expansion policy

Expansion terms should come from an audited domain glossary, a reviewed user
correction, or an offline candidate-generation process. The online path caps
the number of additions. Numeric values, requirement identifiers, signal names,
and version strings should remain unchanged.

Generated multi-query or hypothetical-document expansion can be evaluated as
an optional retrieval arm. It should not replace the original query, and its
results should be labeled so its marginal contribution and false-positive rate
can be measured.

## Feedback lifecycle

Explicit correction is stronger evidence than a click. A production event
should include the query, candidate set, selected result, relevance label,
reason code, document version, user or team scope, and timestamp. Avoid storing
raw sensitive query text when a minimized representation is sufficient.

Feedback moves through controlled stages:

```text
session correction
  -> scoped candidate boost or immediate re-search
  -> reviewed positive and hard-negative example
  -> offline replay on development and frozen holdout sets
  -> shadow deployment
  -> limited rollout
  -> global promotion or rollback
```

Never update global ranking weights from one correction. Require minimum
support, deduplicate correlated events, control for position bias, and retain a
rollback path. Separate `wrong document` from `wrong version`, `access denied`,
`missing document`, and `answer unsupported`; they imply different fixes.

## Evaluation

Freeze test data before tuning. Split by document family, product, project, or
intent so paraphrases of the same source do not cross the boundary. Include:

- exact identifiers and symbols;
- aliases, abbreviations, multilingual terms, and typographical errors;
- numeric values, units, ranges, and version constraints;
- description-only and relationship questions;
- conflicting, stale, inaccessible, and unanswerable cases;
- clean negatives that share many surface terms.

Report retrieval Recall@k, MRR, nDCG, abstention, latency per stage, memory, and
worst-group quality. Evaluate answer factuality and citation entailment
separately from retrieval. Paired comparisons and cluster-aware confidence
intervals are preferable when multiple queries refer to the same source.

## Operational controls

- Preserve document provenance, revision, effective date, and access policy.
- Run regression probes for each retrieval arm and alert on zero contribution.
- Log candidate IDs and ranks without copying confidential content when
  possible.
- Version indexes, glossaries, embedding models, rerankers, and evaluation sets.
- Treat model-generated query text and retrieved documents as untrusted input.
- Provide abstention and human escalation for weak or conflicting evidence.

## Further reading

- [TREC Retrieval-Augmented Generation Track](https://trec-rag.github.io/)
- [IIMAS-RAG at SemEval-2026 Task 8](https://aclanthology.org/2026.semeval-1.345/)
- [TechDocRAG](https://doi.org/10.3390/ai7050161)
