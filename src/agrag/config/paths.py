"""Centralized data paths to keep datasets and artifacts organized."""

from __future__ import annotations

import os
from pathlib import Path


DATA_ROOT = Path(os.getenv("AGRAG_DATA_ROOT", "data"))

# Mock/synthetic datasets
MOCK_DATA_DIR = DATA_ROOT / "mock"
MOCK_DATASET_PATH = MOCK_DATA_DIR / "synthetic_dataset.json"
MOCK_EVAL_DATASET_PATH = MOCK_DATA_DIR / "synthetic_dataset_eval.json"
MOCK_EVAL_QUERIES_PATH = MOCK_DATA_DIR / "eval_queries.json"

# Real/production datasets (Ericsson)
ERICSSON_DATA_DIR = DATA_ROOT / "ericsson"
ERICSSON_DATASET_TEMPLATE_PATH = ERICSSON_DATA_DIR / "dataset_template.json"

# Shared artifacts
BM25_INDEX_PATH = DATA_ROOT / "bm25_index.pkl"
EVAL_SUITES_PATH = DATA_ROOT / "eval_suites.json"
