"""
Self-correcting NL2SQL agent.

Loop: generate -> execute -> if error/empty -> rebuild prompt with error -> retry.
Max retries = 2 (enough to fix typos/hallucinated columns, beyond that diminishing returns).

This is the core differentiator vs vanilla NL2SQL projects.
Track retry rate + recovery rate as resume metrics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from src.agent.executor import ExecResult, execute_sql
from src.data.schema_formatter import build_prompt


# Type alias for any callable that takes prompt and returns SQL string.
# Could be HF pipeline, vLLM, OpenAI client, etc.
GenerateFn = Callable[[str], str]


@dataclass
class AgentTrace:
    """Full trace of one agent run — useful for debugging + eval analysis."""
    question: str
    db_path: str
    attempts: list[dict] = field(default_factory=list)
    final_sql: str = ""
    final_result: ExecResult | None = None
    succeeded: bool = False
    retries_used: int = 0


def build_retry_prompt(
    schema: str,
    question: str,
    failed_sql: str,
    error_or_status: str,
    evidence: str = "",
) -> str:
    """
    Construct retry prompt that gives the model:
      1. Original schema + question
      2. The SQL it just generated
      3. Why it failed
      4. Instruction to fix it
    """
    base = build_prompt(schema, question, evidence)
    # Strip the trailing '### SQL:\n' marker — re-add it after error block
    base_no_tail = base.rsplit("### SQL:", 1)[0]

    retry_block = (
        f"### Previous attempt:\n{failed_sql}\n\n"
        f"### Execution feedback:\n{error_or_status}\n\n"
        f"Fix the SQL. Pay attention to column names, table aliases, and JOIN conditions.\n\n"
        f"### SQL:\n"
    )
    return base_no_tail + retry_block


def run_agent(
    question: str,
    schema: str,
    db_path: str,
    generate_fn: GenerateFn,
    evidence: str = "",
    max_retries: int = 2,
    treat_empty_as_failure: bool = False,
) -> AgentTrace:
    """
    Run the self-correcting agent loop.

    Args:
        question: natural language question
        schema: pre-formatted schema string (from schema_formatter.format_schema)
        db_path: path to sqlite db for execution
        generate_fn: callable that takes prompt str and returns SQL str
        evidence: optional BIRD-style evidence/hint text
        max_retries: number of retries after initial attempt (default 2 = 3 total tries)
        treat_empty_as_failure: if True, empty results trigger retry. Default False
            because many valid queries legitimately return zero rows.

    Returns:
        AgentTrace with full attempt history
    """
    trace = AgentTrace(question=question, db_path=db_path)

    prompt = build_prompt(schema, question, evidence)
    last_result: ExecResult | None = None
    last_sql: str = ""

    for attempt in range(max_retries + 1):
        sql = generate_fn(prompt).strip()
        sql = _clean_generated_sql(sql)
        result = execute_sql(sql, db_path)

        trace.attempts.append({
            "attempt": attempt,
            "sql": sql,
            "status": result.short_summary(),
        })
        last_result = result
        last_sql = sql

        # Success criteria
        is_failure = (not result.success) or (treat_empty_as_failure and result.is_empty)
        if not is_failure:
            trace.succeeded = True
            trace.retries_used = attempt
            break

        # Don't bother building retry prompt if we're out of attempts
        if attempt == max_retries:
            break

        feedback = result.error if not result.success else "Query returned 0 rows. Verify filters and joins."
        prompt = build_retry_prompt(schema, question, sql, feedback or "unknown error", evidence)

    trace.final_sql = last_sql
    trace.final_result = last_result
    trace.retries_used = len(trace.attempts) - 1
    return trace


def _clean_generated_sql(sql: str) -> str:
    """
    Strip common model artifacts:
      - Markdown fences (```sql ... ```)
      - Trailing explanation text after the query
      - Multiple statements (keep only first)
    """
    sql = sql.strip()

    # Strip markdown fences
    if sql.startswith("```"):
        sql = sql.split("\n", 1)[1] if "\n" in sql else sql[3:]
        if sql.endswith("```"):
            sql = sql[:-3]
        sql = sql.strip()
        # Remove leading 'sql' language tag if present
        if sql.lower().startswith("sql\n"):
            sql = sql[4:]

    # Take only first statement (some models add trailing prose)
    # Find first semicolon at end of line
    lines = []
    for line in sql.split("\n"):
        lines.append(line)
        if line.rstrip().endswith(";"):
            break
    sql = "\n".join(lines).strip()

    if not sql.endswith(";"):
        sql += ";"
    return sql


if __name__ == "__main__":
    # Smoke test with a dummy generator
    def dummy_generate(prompt: str) -> str:
        if "Previous attempt" in prompt:
            return "SELECT name FROM customers WHERE country = 'Germany';"
        return "SELECT nme FROM customers WHERE country = 'Germany';"  # typo

    # Requires a real sqlite db — fill in path
    print("import and call run_agent() with your generate fn")
