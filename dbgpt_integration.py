"""
DB-GPT Core Adapter (pluggable, no hard dependency)
- Provides DB-GPT-style SQL candidate generation, parsing, and repair
- Uses sqlglot if available; falls back gracefully when not installed
- Designed to be enabled via feature flag without breaking existing API
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

try:
    # Optional dependency for robust SQL parsing
    from sqlglot import parse_one  # type: ignore
    _SQLGLOT_AVAILABLE = True
except Exception:
    parse_one = None  # type: ignore
    _SQLGLOT_AVAILABLE = False

# NOTE: Real DB-GPT Core APIs could be used here if installed.
# We gate on presence to avoid hard dependency.
_DBGPT_AVAILABLE = False
try:
    import dbgpt_core  # type: ignore  # noqa: F401
    _DBGPT_AVAILABLE = True
except Exception:
    try:
        import dbgpt  # type: ignore  # noqa: F401
        _DBGPT_AVAILABLE = True
    except Exception:
        _DBGPT_AVAILABLE = False


def available() -> bool:
    """Return True if enhanced pipeline can run (sqlglot and/or DB-GPT core present)."""
    return bool(_SQLGLOT_AVAILABLE or _DBGPT_AVAILABLE)


def _try_parse(sql: str) -> bool:
    """Parse SQL using sqlglot if available; be permissive otherwise."""
    if not _SQLGLOT_AVAILABLE:
        # Can't parse strictly; allow and rely on downstream validators
        return bool(sql and sql.strip().lower().startswith("select"))
    try:
        _ = parse_one(sql, read="tsql")
        return True
    except Exception as parse_err:
        logger.debug(f"sqlglot parse failed: {parse_err}")
        return False


def _rank_candidates(candidates: List[str]) -> List[str]:
    # Simple ranking: valid-first, longer queries (more structure) first
    scored: List[Tuple[int, int, str]] = []
    for sql in candidates:
        valid = 1 if _try_parse(sql) else 0
        scored.append((valid, len(sql), sql))
    scored.sort(reverse=True)
    return [s for _, __, s in scored]


def generate_sql_with_dbgpt(
    question: str,
    schema_context: str,
    llm,
    max_candidates: int = 3,
) -> str:
    """Generate multiple SQL candidates and return the first that parses.

    This is a DB-GPT-style approach: k candidates + parser filter. It uses our LLM
    and schema context; if DB-GPT Core is installed later, this function can be
    upgraded to use its join planning and retrieval.
    """
    prompt_base = (
        "You are a T-SQL generator. Return ONLY one SQL statement.\n"
        "Rules: use [SISL Live].[dbo].[Table] aliases, JOIN must have ON, no placeholders.\n"
        "Schema:\n{schema}\n\nQuestion: {q}\n\nSQL:"
    )
    candidates: List[str] = []
    for i in range(max_candidates):
        result = llm.invoke(prompt_base.format(schema=schema_context, q=question))
        sql = str(result.content).strip()
        candidates.append(sql)
    # Keep best
    for sql in _rank_candidates(candidates):
        if _try_parse(sql):
            return sql
    # If none parse, return the first to trigger repair/validation upstream
    return candidates[0] if candidates else ""


def validate_and_repair(
    sql: str,
    schema_context: str,
    llm,
    db_error: Optional[str] = None,
    max_retries: int = 2,
) -> str:
    """Try to repair SQL using model with explicit error feedback."""
    if not sql:
        return sql
    if _try_parse(sql):
        return sql
    last_sql = sql
    for attempt in range(max_retries):
        repair_prompt = (
            "Fix this T-SQL to be valid and executable.\n"
            "- Keep the same intent.\n"
            "- Use [SISL Live].[dbo].[Table] aliases and JOIN ... ON.\n"
            "- No placeholders (e.g., FROM JOIN). Only one statement.\n"
            f"Schema:\n{schema_context}\n\n"
            f"Previous SQL:\n{last_sql}\n\n"
            f"Error (if any):\n{db_error or 'syntax error'}\n\n"
            "Return ONLY the corrected SQL statement."
        )
        result = llm.invoke(repair_prompt)
        fixed = str(result.content).strip()
        if _try_parse(fixed):
            return fixed
        last_sql = fixed
    return last_sql