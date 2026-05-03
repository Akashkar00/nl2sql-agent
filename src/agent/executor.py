"""
SQL executor for SQLite databases.

Every error is captured as structured data so the self-correction loop
can feed it back to the model. Timeouts prevent runaway queries from
blocking inference.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Any


@dataclass
class ExecResult:
    success: bool
    rows: list[tuple] | None = None
    columns: list[str] | None = None
    error: str | None = None
    sql: str = ""

    @property
    def is_empty(self) -> bool:
        """Empty result is suspicious — often a sign of wrong filter."""
        return self.success and (self.rows is None or len(self.rows) == 0)

    def short_summary(self) -> str:
        if not self.success:
            return f"ERROR: {self.error}"
        if self.is_empty:
            return "EMPTY result (0 rows)"
        return f"OK ({len(self.rows)} rows)"


def execute_sql(
    sql: str,
    db_path: str,
    timeout_seconds: float = 10.0,
    max_rows: int = 1000,
) -> ExecResult:
    """Execute SQL on a SQLite DB. Returns ExecResult with success/error/rows."""
    sql = sql.strip().rstrip(";").strip() + ";"

    conn: sqlite3.Connection | None = None
    try:
        conn = sqlite3.connect(db_path, timeout=timeout_seconds)
        conn.text_factory = lambda b: b.decode(errors="ignore")
        cur = conn.cursor()
        cur.execute(sql)
        rows: list[tuple[Any, ...]] = cur.fetchmany(max_rows)
        columns = [d[0] for d in cur.description] if cur.description else []
        return ExecResult(success=True, rows=rows, columns=columns, sql=sql)
    except sqlite3.Error as e:
        return ExecResult(success=False, error=str(e), sql=sql)
    except Exception as e:
        return ExecResult(success=False, error=f"unexpected: {e}", sql=sql)
    finally:
        if conn is not None:
            conn.close()


def results_equal(a: ExecResult, b: ExecResult, order_matters: bool = False) -> bool:
    """
    Compare two execution results for execution-accuracy evaluation.

    Default: set comparison (order-independent), which matches Spider/BIRD eval.
    Set order_matters=True for ORDER BY queries if you want strict comparison.
    """
    if not (a.success and b.success):
        return False
    if a.rows is None or b.rows is None:
        return a.rows == b.rows
    if order_matters:
        return a.rows == b.rows
    # Set comparison via tuple-of-tuples (handles None, ints, strings)
    return set(map(tuple, a.rows)) == set(map(tuple, b.rows))


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3:
        result = execute_sql(sys.argv[2], sys.argv[1])
        print(result.short_summary())
        if result.success and result.rows:
            for row in result.rows[:5]:
                print(row)
    else:
        print("usage: python executor.py db.sqlite 'SELECT * FROM table'")
