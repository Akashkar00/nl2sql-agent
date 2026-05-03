"""
Schema formatting for NL2SQL prompts.

Format choice matters more than hyperparameters. The CREATE TABLE + sample rows
format ('M-Schema' style) consistently beats raw column lists in benchmarks.

Reference: BIRD paper uses similar schema serialization.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass


@dataclass
class Column:
    name: str
    dtype: str
    is_pk: bool = False


@dataclass
class ForeignKey:
    from_table: str
    from_col: str
    to_table: str
    to_col: str


def get_tables(conn: sqlite3.Connection) -> list[str]:
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    )
    return [row[0] for row in cur.fetchall()]


def get_columns(conn: sqlite3.Connection, table: str) -> list[Column]:
    # PRAGMA returns: cid, name, type, notnull, dflt_value, pk
    cur = conn.execute(f"PRAGMA table_info('{table}')")
    return [Column(name=row[1], dtype=row[2] or "TEXT", is_pk=bool(row[5])) for row in cur.fetchall()]


def get_foreign_keys(conn: sqlite3.Connection, table: str) -> list[ForeignKey]:
    cur = conn.execute(f"PRAGMA foreign_key_list('{table}')")
    # PRAGMA returns: id, seq, table, from, to, on_update, on_delete, match
    return [
        ForeignKey(from_table=table, from_col=row[3], to_table=row[2], to_col=row[4])
        for row in cur.fetchall()
    ]


def get_sample_rows(conn: sqlite3.Connection, table: str, n: int = 3) -> list[tuple]:
    try:
        cur = conn.execute(f"SELECT * FROM '{table}' LIMIT {n}")
        return cur.fetchall()
    except sqlite3.Error:
        return []


def format_schema(db_path: str, include_samples: bool = True, sample_rows: int = 3) -> str:
    """
    Build a CREATE TABLE-style schema string with optional sample rows.

    Returns text suitable for direct injection into the model prompt.
    """
    conn = sqlite3.connect(db_path)
    conn.text_factory = lambda b: b.decode(errors="ignore")

    parts: list[str] = []
    tables = get_tables(conn)

    for table in tables:
        cols = get_columns(conn, table)
        fks = get_foreign_keys(conn, table)

        col_defs = []
        for c in cols:
            line = f"    {c.name} {c.dtype}"
            if c.is_pk:
                line += " PRIMARY KEY"
            col_defs.append(line)

        for fk in fks:
            col_defs.append(
                f"    FOREIGN KEY ({fk.from_col}) REFERENCES {fk.to_table}({fk.to_col})"
            )

        ddl = f"CREATE TABLE {table} (\n" + ",\n".join(col_defs) + "\n);"
        parts.append(ddl)

        if include_samples:
            rows = get_sample_rows(conn, table, sample_rows)
            if rows:
                col_names = ", ".join(c.name for c in cols)
                parts.append(f"-- Sample rows from {table} ({col_names}):")
                for row in rows:
                    formatted = ", ".join(repr(v) for v in row)
                    parts.append(f"-- ({formatted})")
        parts.append("")  # blank line between tables

    conn.close()
    return "\n".join(parts).strip()


def build_prompt(schema: str, question: str, evidence: str = "") -> str:
    """Build the inference-time prompt. Match this EXACTLY at training time."""
    evidence_block = f"\n### Evidence:\n{evidence}\n" if evidence else ""
    return (
        f"### Schema:\n{schema}\n"
        f"{evidence_block}"
        f"### Question:\n{question}\n\n"
        f"### SQL:\n"
    )


if __name__ == "__main__":
    # Smoke test against any sqlite file
    import sys
    if len(sys.argv) > 1:
        print(format_schema(sys.argv[1]))
    else:
        print("usage: python schema_formatter.py path/to/db.sqlite")
