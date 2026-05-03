"""
Execution accuracy: does predicted SQL produce the same rows as gold SQL?

This is the metric Spider and BIRD use. Exact-match is broken because there
are many valid SQL forms for the same question (different aliases, JOIN orders,
subquery vs CTE, etc.).
"""
from __future__ import annotations

from src.agent.executor import execute_sql, results_equal


def execution_accuracy(
    pred_sql: str,
    gold_sql: str,
    db_path: str,
    order_matters: bool = False,
) -> bool:
    """
    Returns True if pred_sql and gold_sql produce equivalent results when executed.

    order_matters: set True if the gold SQL contains ORDER BY (Spider eval does this).
                   Default False is more lenient and faster to evaluate.
    """
    if "order by" in gold_sql.lower():
        order_matters = True

    pred = execute_sql(pred_sql, db_path)
    gold = execute_sql(gold_sql, db_path)

    if not gold.success:
        # Gold SQL itself broken — should not happen in clean datasets, but log it
        return False

    return results_equal(pred, gold, order_matters=order_matters)


def categorize_failure(pred_sql: str, gold_sql: str, db_path: str) -> str:
    """Bucket failures for error analysis. Use for the resume table."""
    pred = execute_sql(pred_sql, db_path)
    gold = execute_sql(gold_sql, db_path)

    if not pred.success:
        err = (pred.error or "").lower()
        if "no such column" in err:
            return "hallucinated_column"
        if "no such table" in err:
            return "hallucinated_table"
        if "syntax error" in err:
            return "syntax_error"
        if "ambiguous" in err:
            return "ambiguous_column"
        return "other_execution_error"

    if pred.is_empty and not gold.is_empty:
        return "wrong_filter_or_join"

    if not gold.is_empty and pred.rows and len(pred.rows) != len(gold.rows):
        return "wrong_aggregation_or_distinct"

    return "wrong_values"
