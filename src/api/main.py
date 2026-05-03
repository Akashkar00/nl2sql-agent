"""
FastAPI service exposing the NL2SQL agent.

Run:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000

Endpoints:
    POST /query           -> generate SQL with self-correction
    GET  /health          -> liveness check
    GET  /databases       -> list available DBs
"""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.agent.executor import execute_sql
from src.agent.self_correction import run_agent
from src.data.schema_formatter import format_schema
from src.model.inference import get_engine


# ---- Config ----
DB_ROOT = Path(os.getenv("DB_ROOT", "data/spider/database"))


# ---- Schemas ----
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    db_id: str = Field(..., description="database identifier matching a folder under DB_ROOT")
    evidence: str = Field("", description="optional BIRD-style hint")
    use_agent: bool = True
    max_retries: int = Field(2, ge=0, le=5)


class Attempt(BaseModel):
    attempt: int
    sql: str
    status: str


class QueryResponse(BaseModel):
    sql: str
    success: bool
    rows: list[list] | None
    columns: list[str] | None
    error: str | None
    retries_used: int
    attempts: list[Attempt]


# ---- App ----
app = FastAPI(title="NL2SQL Agent", version="0.1.0")


def _resolve_db_path(db_id: str) -> Path:
    candidates = [
        DB_ROOT / db_id / f"{db_id}.sqlite",
        DB_ROOT / f"{db_id}.sqlite",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise HTTPException(status_code=404, detail=f"db_id '{db_id}' not found under {DB_ROOT}")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/databases")
def list_databases():
    if not DB_ROOT.exists():
        return {"databases": [], "db_root": str(DB_ROOT)}
    dbs = sorted([p.name for p in DB_ROOT.iterdir() if p.is_dir()])
    return {"databases": dbs, "db_root": str(DB_ROOT)}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    db_path = _resolve_db_path(req.db_id)
    schema = format_schema(str(db_path))

    engine = get_engine()

    def generate_fn(prompt: str) -> str:
        return engine.generate(prompt, max_new_tokens=256)

    if req.use_agent:
        trace = run_agent(
            question=req.question,
            schema=schema,
            db_path=str(db_path),
            generate_fn=generate_fn,
            evidence=req.evidence,
            max_retries=req.max_retries,
        )
        result = trace.final_result
        return QueryResponse(
            sql=trace.final_sql,
            success=result.success if result else False,
            rows=[list(r) for r in (result.rows or [])] if result and result.success else None,
            columns=result.columns if result and result.success else None,
            error=result.error if result and not result.success else None,
            retries_used=trace.retries_used,
            attempts=[Attempt(**a) for a in trace.attempts],
        )
    else:
        from src.agent.self_correction import _clean_generated_sql
        from src.data.schema_formatter import build_prompt
        prompt = build_prompt(schema, req.question, req.evidence)
        sql = _clean_generated_sql(generate_fn(prompt))
        result = execute_sql(sql, str(db_path))
        return QueryResponse(
            sql=sql,
            success=result.success,
            rows=[list(r) for r in (result.rows or [])] if result.success else None,
            columns=result.columns if result.success else None,
            error=result.error if not result.success else None,
            retries_used=0,
            attempts=[Attempt(attempt=0, sql=sql, status=result.short_summary())],
        )
