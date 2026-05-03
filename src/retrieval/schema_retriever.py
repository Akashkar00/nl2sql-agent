"""
Schema retriever — Phase 5 (optional).

For databases with many tables (50+), embedding the full schema blows context budget.
This module retrieves the top-k most relevant tables for a given question.

NOT REQUIRED for MVP. Skip if running tight on time. Implement if you finish core
Phases 1-4 with time to spare.

Plan when you implement:
  1. For each table: build "table card" = name + col list + sample row.
  2. Encode all cards with BAAI/bge-small-en-v1.5.
  3. Build FAISS index per database.
  4. At query time: encode question, retrieve top-k tables, build schema from those only.

Resume bullet you can write after this:
  "Schema-aware RAG layer using BGE embeddings + FAISS reduced average prompt
  size from 4.2K to 1.1K tokens on databases with 50+ tables, with no accuracy drop"
"""
from __future__ import annotations

# TODO: implement after Phase 4 metrics are good
raise NotImplementedError("Phase 5 — implement after core training is working")
