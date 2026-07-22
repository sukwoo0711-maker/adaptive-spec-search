# Porting decisions required

Resolve these items in the deployment repository. They are intentionally not
guessed here.

- `PORTING-DECISION-AUTHZ`: where caller identity and document authorization are
  enforced before indexing and retrieval.
- `PORTING-DECISION-EMBEDDING`: local or hosted embedding implementation, plus
  data retention, region, model/version pinning, and outage behavior.
- `PORTING-DECISION-LIMITS`: maximum documents, characters per document,
  candidate count, query length, concurrency, memory, and latency budget.
- `PORTING-DECISION-PARSING`: supported file types and explicit coverage for
  tables, merged cells, formulas, images, OCR, revisions, and attachments.
- `PORTING-DECISION-ABSTENTION`: minimum evidence policy and the user-visible
  distinction among not found, filtered, stale index, parser failure, and access
  denied.
- `PORTING-DECISION-FEEDBACK`: identity, scope, expiry, encryption, deletion,
  review threshold, rollback, and audit retention for feedback events.
- `PORTING-DECISION-EVALUATION`: frozen corpus/query splits, metrics, subgroup
  gates, security tests, and promotion/rollback thresholds.
