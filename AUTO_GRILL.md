# Auto-grill review

Review date: 2026-07-22

## Claims that survive review

- Reciprocal Rank Fusion combines rankings; it does not calibrate lexical and
  dense scores or prove relevance.
- Audited synonym expansion can help vocabulary mismatch, but can also reduce
  precision. The implementation caps additions and preserves the original query.
- An empty result means only that this configured pipeline found no eligible
  candidate. It does **not** prove that the specification contains no answer.
- Search scores are ordering signals, not probabilities or confidence values.

## Failure modes tested

- Duplicate document identifiers are rejected because they break provenance.
- Empty queries abstain, invalid limits fail closed, and document/text bounds are
  configurable.
- Embedding row count, dimensions, and finite values are validated.
- Metadata constraints are applied to candidate eligibility before rank fusion
  and dense scoring.
- Feedback may adjust only an already-retrieved candidate and is bounded.

## Security and privacy boundary

This library performs no network requests itself. An injected `embed` callable
is executable code and may send every indexed document and query elsewhere.
Use a trusted local implementation for confidential material, or document and
approve the provider, retention, region, and access policy. Metadata filtering
does not undo disclosure that already occurred when document embeddings were
created. Authorization must therefore also be enforced before documents are
passed to this engine.

Treat document text, metadata, queries, synonyms, embeddings, feedback, and
logs as sensitive inputs. Do not log raw values by default. The package does not
parse files, follow links, persist an index, authenticate callers, or provide a
tenant boundary.

## Remaining limits

- The tokenizer is intentionally small and is not a complete Korean, Unicode,
  identifier, formula, table, image, or OCR parser.
- Candidate filtering in this in-memory example is not a substitute for
  storage-layer row-level security.
- No claim of completeness is possible without a versioned corpus manifest,
  parser coverage report, access-policy audit, and an evaluation set containing
  unanswerable and restricted cases.
- Dense embeddings are held in memory and brute-force searched; this is a
  reference implementation, not a capacity claim.
