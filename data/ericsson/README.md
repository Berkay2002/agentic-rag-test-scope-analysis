# Ericsson Data Drop (Placeholder)

Place the real Ericsson export(s) in this folder and convert them into the
standard ingestion schema used by `agrag ingest`.

Expected dataset shape (see `dataset_template.json`):
- `metadata`: source_system, schema_version, generated_at
- `entities`: list of normalized entities (Requirement, TestCase, Function, etc.)
- `relationships`: list of edges with `relationship_type`, `source_id`, `target_id`

Keep raw vendor files in a subfolder if needed (for example `raw/`).
