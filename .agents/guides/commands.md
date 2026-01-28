# Commands

## Setup
```bash
poetry install
cp .env.example .env
poetry run agrag init
```

## Run the Agent
```bash
# Interactive chat (safe mode)
poetry run agrag chat
poetry run agrag chat --thread-id my-session

# YOLO mode (autonomous execution)
poetry run agrag chat --yolo

# Headless
poetry run agrag -p "query here" --output-format json
poetry run agrag query "your question" --stream
poetry run agrag info
```

## Data Generation + Ingestion
```bash
poetry run agrag generate --requirements 50 --testcases 200
poetry run agrag ingest data/mock/synthetic_dataset.json
```

## Load External Data
```bash
# Documentation (Docling)
poetry run agrag load docs /path/to/docs --formats pdf,docx --use-chunker

# Code repository (AST parsing)
poetry run agrag load repo /path/to/repo --languages python,java

# Ericsson TGF CSV
poetry run agrag load tgf /path/to/tgf_export.csv
```

## Evaluation
```bash
poetry run agrag evaluate \
  --dataset data/mock/eval_queries.json \
  --output results.json \
  --k-values "1,3,5,10"
```

## Logging
```bash
poetry run agrag --log-level DEBUG query "..."
poetry run agrag --log-format json query "..."
```
