from __future__ import annotations

# Standalone DB-GPT-style RAG API (no env flags)
# - Dynamic SQL generation with k-candidate strategy
# - Schema-aware validation and lightweight repair
# - Optional sqlglot parsing if installed
# - Streaming and non-streaming endpoints

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, AsyncGenerator, Tuple

import asyncio
import logging
import json
import threading
import re

import pyodbc
from langchain_qdrant import QdrantVectorStore
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document

# Optional parser for robust SQL structure checks
try:
    from sqlglot import parse_one  # type: ignore
    _SQLGLOT_AVAILABLE = True
except Exception:
    parse_one = None  # type: ignore
    _SQLGLOT_AVAILABLE = False

# -----------------------------
# Configuration (no env vars)
# -----------------------------
QDRANT_URL = "http://192.168.0.120:6333/"
COLLECTION_NAME = "schema_vectors"

MSSQL_CONN_STR = (
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=192.168.1.44,1433;"
    "DATABASE=SISL Live;"
    "UID=test;"
    "PWD=Test@345;"
    "TrustServerCertificate=yes;"
)

OLLAMA_BASE = "http://192.168.0.120:11434"
OLLAMA_EMBED_MODEL = "nomic-embed-text"
OLLAMA_CHAT_MODEL = "mistral:latest"

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api_dbgpt")

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(
    title="SISL RAG API (DB-GPT Style)",
    description="DB-GPT-style candidate/validate/repair SQL generation",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Models
# -----------------------------
class QueryRequest(BaseModel):
    question: str
    chat_id: Optional[str] = None
    max_results: Optional[int] = 5

class QueryResponse(BaseModel):
    question: str
    generated_sql: str
    results: List[Dict[str, Any]]
    columns: List[str]
    row_count: int
    success: bool
    error: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    message: str

class StreamingQueryRequest(BaseModel):
    question: str
    chat_id: Optional[str] = None
    max_results: Optional[int] = 5
    stream: bool = True

class StreamingChunk(BaseModel):
    type: str
    data: Dict[str, Any]
    timestamp: float

# -----------------------------
# Globals
# -----------------------------
qdrant: QdrantVectorStore | None = None
embeddings: OllamaEmbeddings | None = None
llm: ChatOllama | None = None
db_schema: Dict[str, Any] | None = None

# Conversation memory
class ChatMemoryManager:
    def __init__(self):
        self.memories: Dict[str, ConversationBufferMemory] = {}
        self.lock = threading.Lock()

    def get_memory(self, chat_id: str) -> ConversationBufferMemory:
        with self.lock:
            if chat_id not in self.memories:
                self.memories[chat_id] = ConversationBufferMemory(
                    memory_key="chat_history", return_messages=True
                )
            return self.memories[chat_id]

memory_manager = ChatMemoryManager()

# -----------------------------
# Utilities: SQL handling
# -----------------------------

def extract_sql_from_response(response: str) -> str:
    """Extract the FIRST valid-looking SELECT statement from an LLM response.

    Falls back to ERROR_NOT_FOUND if no SELECT is present, to trigger regeneration.
    """
    try:
        text = response.replace("```sql", "").replace("```", "").strip()
        for lead in ("SQL:", "T-SQL:", "TSql:", "Query:"):
            if text.startswith(lead):
                text = text[len(lead):].strip()
        m = re.search(r"(?is)\bselect\b.*?;", text)
        if m:
            return " ".join(m.group(0).split())
        m2 = re.search(r"(?is)\bselect\b.*", text)
        if m2:
            candidate = " ".join(m2.group(0).split())
            if not candidate.endswith(";"):
                candidate += ";"
            return candidate
        return "ERROR_NOT_FOUND"
    except Exception:
        return "ERROR_NOT_FOUND"


def repair_sql_common_issues(sql_query: str) -> str:
    """Light repairs for common LLM artifacts."""
    try:
        sql = " ".join(sql_query.split())
        sql = sql.replace(", FROM", " FROM").replace(",  FROM", " FROM")
        sql = sql.replace(" FROM JOIN ", " FROM ")
        sql = " ".join(sql.split())
        return sql
    except Exception:
        return sql_query


def _try_parse(sql: str) -> bool:
    if not sql or not sql.strip().lower().startswith("select"):
        return False
    if not _SQLGLOT_AVAILABLE:
        return True
    try:
        _ = parse_one(sql, read="tsql")
        return True
    except Exception:
        return False


def validate_sql_schema_aware(sql_query: str, schema_info: Dict[str, Any]) -> str:
    sql_cleaned = " ".join(sql_query.split())
    if not sql_cleaned.upper().startswith("SELECT"):
        raise ValueError("Query must start with SELECT")
    if sql_cleaned.count("(") != sql_cleaned.count(")"):
        raise ValueError("Mismatched parentheses in SQL")

    upper_sql = sql_cleaned.upper()
    if " FROM JOIN " in upper_sql or upper_sql.startswith("SELECT ,"):
        raise ValueError("Malformed FROM/JOIN structure detected")

    # JOIN syntax checks
    if "JOIN" in upper_sql:
        join_pattern = r"JOIN\s+\[[^\]]+\]\.\[[^\]]+\]\.\[[^\]]+\](?:\s+\w+)?\s+ON"
        if not re.search(join_pattern, sql_cleaned, re.IGNORECASE):
            if "JOIN" in upper_sql and "ON" not in upper_sql:
                raise ValueError("JOIN clause missing ON condition")
        if upper_sql.count(" JOIN ") > 0 and upper_sql.count(" ON ") < upper_sql.count(" JOIN "):
            raise ValueError("At least one JOIN is missing an ON condition")

    # Validate referenced tables appear in schema; attempt fuzzy repair if needed
    table_pattern = r"\[SISL Live\]\.\[dbo\]\.\[([^\]]+)\]"
    table_matches = re.findall(table_pattern, sql_cleaned)
    schema_tables = list(schema_info["tables"].keys())

    def _norm_table(s: str) -> List[str]:
        # Tokenize by non-alnum and keep tokens
        return [tok for tok in re.split(r"[^a-z0-9]+", s.lower()) if tok]

    for table in table_matches:
        if table in schema_info["tables"]:
            continue
        # Fuzzy candidate by token overlap and substring
        t_tokens = set(_norm_table(table))
        best = None  # (score, candidate)
        for cand in schema_tables:
            c_tokens = set(_norm_table(cand))
            overlap = len(t_tokens & c_tokens)
            substr = 1 if (table.lower() in cand.lower() or cand.lower() in table.lower()) else 0
            score = overlap * 10 + substr * 5
            if best is None or score > best[0]:
                best = (score, cand)
        if best and best[0] >= 10:
            sql_cleaned = sql_cleaned.replace(f"[{table}]", f"[{best[1]}]")
        else:
            raise ValueError(f"Table '{table}' not found in schema")

    # Alias checks
    from_matches = re.findall(r"FROM\s+\[SISL Live\]\.\[dbo\]\.\[([^\]]+)\](?:\s+(\w+))?", sql_cleaned, re.IGNORECASE)
    join_matches = re.findall(r"JOIN\s+\[SISL Live\]\.\[dbo\]\.\[([^\]]+)\](?:\s+(\w+))?", sql_cleaned, re.IGNORECASE)
    defined_aliases = {a.lower() for _, a in from_matches if a} | {a.lower() for _, a in join_matches if a}
    alias_to_table: Dict[str, str] = {}
    for t, a in from_matches:
        if a:
            alias_to_table[a.lower()] = t
    for t, a in join_matches:
        if a:
            alias_to_table[a.lower()] = t
    used_aliases = {a.lower() for a in re.findall(r"(\w+)\.\[", sql_cleaned)}
    undefined = used_aliases - defined_aliases
    if undefined:
        raise ValueError(f"SQL contains undefined table aliases: {undefined}")

    # Column existence checks with light fuzzy repair
    def _norm(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", s.lower())

    alias_col_refs = re.findall(r"(\w+)\.\[([^\]]+)\]", sql_cleaned)
    for alias, col in alias_col_refs:
        a = alias.lower()
        table = alias_to_table.get(a)
        if not table:
            continue
        table_meta = schema_info["tables"].get(table)
        if not table_meta:
            continue
        cols = [c["name"] for c in table_meta.get("columns", [])]
        if col in cols:
            continue
        # Fuzzy match within the same table
        target = _norm(col)
        best_name = None
        best_score = -1
        for c in cols:
            n = _norm(c)
            score = 0
            if n == target:
                score = 100
            elif target and (target in n or n in target):
                score = 60
            # token overlap
            tokens_t = set(re.split(r"[^a-z0-9]+", target))
            tokens_c = set(re.split(r"[^a-z0-9]+", n))
            score += len(tokens_t & tokens_c) * 10
            if score > best_score:
                best_score = score
                best_name = c
        if best_name and best_score >= 60:
            # Replace alias.[col] with alias.[best_name]
            pattern = rf"\b{re.escape(alias)}\.\[{re.escape(col)}\]"
            replacement = f"{alias}.[{best_name}]"
            sql_cleaned = re.sub(pattern, replacement, sql_cleaned)
        else:
            raise ValueError(f"Column '{col}' not found in table '{table}'")

    return sql_cleaned

# -----------------------------
# Schema discovery and context
# -----------------------------

def _is_excluded_table(table_name: str) -> bool:
    """Heuristically exclude system/security/admin tables (no hardcoded business tables)."""
    name = table_name.lower().strip()
    # Common NAV/SQL admin/security/system tables or prefixes
    exclude_substrings = [
        "$ndo$",  # NAV system
        "access control",
        "active session",
        "permission",
        "user",
        "role",
        "license",
        "change log",
        "aggregation",
        "_permission",
    ]
    return any(s in name for s in exclude_substrings)

def _is_config_table(table_name: str) -> bool:
    """Heuristic: configuration/lookup tables we should avoid for analytics."""
    name = table_name.lower().strip()
    config_substrings = [
        "setup", "code", "chart", "config", "dimension", "template", "lookup"
    ]
    return any(s in name for s in config_substrings)


def discover_database_schema() -> Dict[str, Any]:
    conn = pyodbc.connect(MSSQL_CONN_STR)
    cursor = conn.cursor()
    schema_info: Dict[str, Any] = {"tables": {}, "relationships": [], "column_details": {}}

    cursor.execute(
        """
        SELECT TABLE_SCHEMA, TABLE_NAME, TABLE_TYPE
        FROM INFORMATION_SCHEMA.TABLES
        WHERE TABLE_TYPE='BASE TABLE'
        ORDER BY TABLE_NAME
        """
    )
    for schema, table_name, _ in cursor.fetchall():
        if _is_excluded_table(table_name):
            continue
        schema_info["tables"][table_name] = {"schema": schema, "columns": []}
        cursor.execute(
            """
            SELECT COLUMN_NAME, DATA_TYPE
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_NAME = ?
            ORDER BY ORDINAL_POSITION
            """,
            table_name,
        )
        schema_info["tables"][table_name]["columns"] = [
            {"name": c, "type": t} for c, t in cursor.fetchall()
        ]

    # relationships (FKs)
    cursor.execute(
        """
        SELECT 
          fk.name AS FK_NAME,
          tp.name AS PARENT_TABLE,
          tr.name AS REF_TABLE,
          cp.name AS PARENT_COLUMN,
          cr.name AS REF_COLUMN
        FROM sys.foreign_keys fk
        INNER JOIN sys.foreign_key_columns fkc ON fk.object_id = fkc.constraint_object_id
        INNER JOIN sys.tables tp ON fkc.parent_object_id = tp.object_id
        INNER JOIN sys.columns cp ON fkc.parent_object_id = cp.object_id AND fkc.parent_column_id = cp.column_id
        INNER JOIN sys.tables tr ON fkc.referenced_object_id = tr.object_id
        INNER JOIN sys.columns cr ON fkc.referenced_object_id = cr.object_id AND fkc.referenced_column_id = cr.column_id
        """
    )
    for _, parent, ref, pcol, rcol in cursor.fetchall():
        schema_info["relationships"].append(
            {
                "source_table": parent,
                "source_column": pcol,
                "target_table": ref,
                "target_column": rcol,
            }
        )

    cursor.close()
    conn.close()
    return schema_info


def _build_schema_documents(schema_info: Dict[str, Any]) -> List[Document]:
    """Convert discovered schema to Qdrant-ingestable documents (tables + columns)."""
    docs: List[Document] = []
    for table, meta in schema_info["tables"].items():
        cols = meta.get("columns", [])
        col_list = ", ".join(f"[{c['name']}] ({c['type']})" for c in cols[:50])
        content = (
            f"TABLE [SISL Live].[dbo].[{table}]\n"
            f"COLUMNS: {col_list}\n"
        )
        docs.append(Document(page_content=content, metadata={
            "type": "table",
            "table_name": table,
        }))
        # Column-level documents
        for c in cols:
            c_name = c["name"]
            c_type = c["type"]
            docs.append(Document(
                page_content=(
                    f"COLUMN [SISL Live].[dbo].[{table}].[{c_name}]\nTYPE: {c_type}"
                ),
                metadata={
                    "type": "column",
                    "table_name": table,
                    "column_name": c_name,
                    "data_type": c_type,
                },
            ))
    return docs


def _ensure_qdrant_index(schema_info: Dict[str, Any]) -> None:
    """If Qdrant collection is empty or missing, (re)index schema documents."""
    global qdrant
    if qdrant is None:
        return
    try:
        probe = qdrant.similarity_search("table", k=1)
        if not probe:
            docs = _build_schema_documents(schema_info)
            if docs:
                qdrant.add_documents(docs)
                logger.info("Qdrant schema indexed (tables + columns)")
    except Exception as e:
        try:
            docs = _build_schema_documents(schema_info)
            if docs:
                qdrant.add_documents(docs)
                logger.info("Qdrant schema indexed after exception: %s", e)
        except Exception as e2:
            logger.warning(f"Failed to index schema in Qdrant: {e2}")


def _norm_token(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def _collect_candidate_columns(schema_info: Dict[str, Any]) -> Dict[str, List[str]]:
    """Collect generic candidate columns by pattern (no hardcoded business naming)."""
    candidates: Dict[str, List[str]] = {
        "customer_keys": [],
        "amount_cols": [],
        "date_cols": [],
    }
    for table, meta in schema_info["tables"].items():
        for col in meta.get("columns", []):
            name = col["name"]
            n = _norm_token(name)
            if ("customer" in n and "no" in n) or ("sell" in n and "customer" in n and "no" in n):
                candidates["customer_keys"].append(f"{table}::{name}")
            if "amount" in n:
                candidates["amount_cols"].append(f"{table}::{name}")
            if "date" in n:
                candidates["date_cols"].append(f"{table}::{name}")
    return candidates


def generate_schema_context(question: str, schema_info: Dict[str, Any], k: int = 8) -> Tuple[str, List[str], Dict[str, List[str]]]:
    """Build a focused schema context and propose candidate base tables.

    Uses keyword heuristics (no hardcoded business tables) to improve ranking.
    """
    q = question.lower()
    # Generic business keywords and stems
    keyword_groups: Dict[str, List[str]] = {
        "customer": ["customer", "customers", "cust", "client", "clients"],
        "invoice": ["invoice", "invoices", "inv"],
        "sales": ["sale", "sales", "revenue"],
        "vendor": ["vendor", "vendors", "supplier", "suppliers"],
        "item": ["item", "items", "product", "products"],
        "ledger": ["ledger", "entry", "entries"],
        "payment": ["payment", "payments", "paid", "receive", "receipt"],
        "bank": ["bank", "cheque", "check"],
        "date": ["date", "period", "quarter", "month", "year"],
    }

    # Flatten terms present in question
    active_terms: List[str] = []
    for terms in keyword_groups.values():
        for t in terms:
            if t in q:
                active_terms.append(t)
    
    tables_scored: List[Tuple[int, str]] = []
    for t, meta in schema_info["tables"].items():
        t_lower = t.lower()
        score = 0
        # Direct mention of table name
        if t_lower in q:
            score += 3
        # Keyword stems vs table name
        for group, terms in keyword_groups.items():
            if any(term in q for term in terms):
                # Boost if the table name hints the same group (generic heuristic)
                if group in t_lower or any(stem in t_lower for stem in terms):
                    score += 3
        # Column-name overlap
        for col in meta.get("columns", []):
            cname = col["name"].lower()
            if cname in q:
                score += 1
        # Penalize configuration/lookup tables
        if _is_config_table(t):
            score -= 3
        # Keep slight baseline for all business tables
        if score > 0:
            tables_scored.append((score, t))

    # Top-k candidate tables, fallback to first k if none scored
    tables_ranked = [t for _, t in sorted(tables_scored, key=lambda x: (-x[0], x[1]))][:k]
    if not tables_ranked:
        tables_ranked = list(schema_info["tables"].keys())[:k]

    # Build context
    parts = ["SCHEMA OVERVIEW (FOCUSED):"]
    for t in tables_ranked:
        cols = schema_info["tables"][t]["columns"]
        col_list = ", ".join(f"[{c['name']}]" for c in cols[:20])
        parts.append(f"- [SISL Live].[dbo].[{t}] cols: {col_list}")
    # Prefer Qdrant retrieval for focused context
    if qdrant is not None:
        try:
            hits = qdrant.similarity_search(question, k=20)
            if not hits:
                _ensure_qdrant_index(schema_info)
                hits = qdrant.similarity_search(question, k=20)
            if hits:
                ctx_lines: List[str] = ["SCHEMA OVERVIEW (RETRIEVED):"]
                candidate_tables: List[str] = []
                col_hints: Dict[str, List[str]] = {"customer_keys": [], "amount_cols": [], "date_cols": []}
                for d in hits[:20]:
                    ctx_lines.append(d.page_content[:500])
                    meta = d.metadata or {}
                    if meta.get("type") == "table":
                        t = meta.get("table_name")
                        if t and t not in candidate_tables and not _is_config_table(t):
                            candidate_tables.append(t)
                    elif meta.get("type") == "column":
                        t = meta.get("table_name")
                        c = meta.get("column_name")
                        if t and c:
                            n = _norm_token(c)
                            qual = f"{t}::{c}"
                            if ("customer" in n and "no" in n) or ("sell" in n and "customer" in n and "no" in n):
                                col_hints["customer_keys"].append(qual)
                            if "amount" in n:
                                col_hints["amount_cols"].append(qual)
                            if "date" in n:
                                col_hints["date_cols"].append(qual)
                if not candidate_tables:
                    candidate_tables = tables_ranked
                return "\n".join(ctx_lines), candidate_tables[:k], col_hints
        except Exception as e:
            logger.info(f"Qdrant retrieval failed, falling back to heuristic: {e}")

    # Heuristic fallback (no Qdrant or no hits)
    col_candidates = _collect_candidate_columns(schema_info)
    context = "\n".join(parts)
    return context, tables_ranked, col_candidates

# -----------------------------
# Candidate generation & repair
# -----------------------------

def _rank_candidates(candidates: List[str]) -> List[str]:
    scored: List[tuple[int, int, str]] = []
    for sql in candidates:
        valid = 1 if _try_parse(sql) else 0
        scored.append((valid, len(sql), sql))
    scored.sort(reverse=True)
    return [s for _, __, s in scored]


def generate_sql_candidates(
    question: str,
    schema_context: str,
    k: int,
    llm: ChatOllama,
    preferred_tables: Optional[List[str]] = None,
    column_hints: Optional[Dict[str, List[str]]] = None,
    allowed_tables: Optional[List[str]] = None,
) -> List[str]:
    prompt = (
        "You are a T-SQL generator. Output ONLY one SQL statement.\n"
        "Rules:\n"
        "- Use [SISL Live].[dbo].[Table] <alias> and square brackets for identifiers.\n"
        "- JOIN must always have ON.\n"
        "- No placeholders like 'FROM JOIN' or dangling commas.\n"
        "- Use ONLY tables listed under 'Allowed Tables'.\n"
        "- One statement only.\n\n"
        f"Schema context:\n{schema_context}\n\n"
        + (
            ("Preferred base tables (choose one if relevant):\n" +
             "\n".join(f"- [SISL Live].[dbo].[{t}]" for t in (preferred_tables or [])) + "\n\n")
            if preferred_tables else ""
        ) +
        (
            "Allowed Tables (use only from this list):\n" +
            ("\n".join(f"- [SISL Live].[dbo].[{t}]" for t in (allowed_tables or [])) + "\n\n")
            if allowed_tables else ""
        ) +
        (
            "Column hints (if applicable, prefer these kinds of columns):\n"
            + ("- Customer key columns: \n" +
               "\n".join(f"  * [{t.split('::',1)[0]}].[{t.split('::',1)[1]}]" for t in (column_hints or {}).get('customer_keys', [])[:6]) + "\n"
               if column_hints and column_hints.get('customer_keys') else "")
            + ("- Amount columns: \n" +
               "\n".join(f"  * [{t.split('::',1)[0]}].[{t.split('::',1)[1]}]" for t in (column_hints or {}).get('amount_cols', [])[:6]) + "\n"
               if column_hints and column_hints.get('amount_cols') else "")
            + ("- Date columns: \n" +
               "\n".join(f"  * [{t.split('::',1)[0]}].[{t.split('::',1)[1]}]" for t in (column_hints or {}).get('date_cols', [])[:6]) + "\n\n"
               if column_hints and column_hints.get('date_cols') else "")
        ) +
        f"Question: {{question}}\n\nSQL:"
    )
    chain = ChatPromptTemplate.from_template(prompt) | llm
    candidates: List[str] = []
    for _ in range(k):
        resp = chain.invoke({"question": question})
        sql_raw = str(resp.content).strip()
        candidates.append(extract_sql_from_response(sql_raw))
    return candidates


def repair_sql(sql: str, schema_context: str, llm: ChatOllama, db_error: str | None = None, max_retries: int = 2) -> str:
    last = sql
    for _ in range(max_retries):
        prompt = (
            "Fix this T-SQL to be valid and executable.\n"
            "- Maintain the same intent.\n"
            "- Use [SISL Live].[dbo].[Table] <alias> and JOIN ... ON.\n"
            "- No placeholders; one statement only.\n\n"
            f"Schema context:\n{schema_context}\n\n"
            f"Previous SQL:\n{last}\n\n"
            f"Error:\n{db_error or 'syntax error'}\n\n"
            "Return ONLY the corrected SQL."
        )
        result = llm.invoke(prompt)
        fixed = extract_sql_from_response(str(result.content).strip())
        if _try_parse(fixed):
            return fixed
        last = fixed
    return last


def _build_fk_graph(schema_info: Dict[str, Any]) -> Dict[str, List[str]]:
    graph: Dict[str, List[str]] = {t: [] for t in schema_info["tables"].keys()}
    for rel in schema_info.get("relationships", []):
        a = rel.get("source_table")
        b = rel.get("target_table")
        if a in graph and b in graph:
            graph[a].append(b)
            graph[b].append(a)
    return graph


def _seed_tables_from_question(question: str, tables: List[str]) -> List[str]:
    q = question.lower()
    seeds: List[str] = []
    domain_tokens = [
        "customer", "cust", "sales", "invoice", "line", "header", "ledger", "entry",
        "vendor", "item", "product"
    ]
    for t in tables:
        tl = t.lower()
        score = sum(1 for tok in domain_tokens if tok in tl and tok in q)
        if score > 0:
            seeds.append(t)
    if not seeds:
        for t in tables:
            if t.lower() in q:
                seeds.append(t)
    seen = set()
    uniq: List[str] = []
    for t in seeds:
        if t not in seen:
            uniq.append(t)
            seen.add(t)
    return uniq[:3] if uniq else tables[:2]


def plan_allowed_tables(question: str, schema_info: Dict[str, Any], candidate_tables: List[str]) -> List[str]:
    graph = _build_fk_graph(schema_info)
    seeds = _seed_tables_from_question(question, candidate_tables or list(graph.keys()))
    allowed: List[str] = []
    seen = set()
    from collections import deque
    dq = deque()
    for s in seeds:
        if s in graph:
            dq.append((s, 0))
            seen.add(s)
            allowed.append(s)
    max_depth = 2
    while dq:
        node, depth = dq.popleft()
        if depth >= max_depth:
            continue
        for nbr in graph.get(node, []):
            if nbr not in seen and not _is_config_table(nbr) and not _is_excluded_table(nbr):
                seen.add(nbr)
                allowed.append(nbr)
                dq.append((nbr, depth + 1))
        if len(allowed) >= 12:
            break
    dedup: List[str] = []
    taken = set()
    for t in allowed:
        if t not in taken:
            dedup.append(t)
            taken.add(t)
    return dedup[:12]


def _enforce_table_whitelist(sql: str, allowed_tables: List[str]) -> None:
    if not allowed_tables:
        return
    tbls = re.findall(r"\[SISL Live\]\.\[dbo\]\.\[([^\]]+)\]", sql)
    extras = [t for t in set(tbls) if t not in set(allowed_tables)]
    if extras:
        raise ValueError(f"SQL uses tables outside allowed set: {extras}")


def generate_sql_dbgpt_style(question: str, schema_info: Dict[str, Any]) -> str:
    assert llm is not None
    schema_context, candidate_tables, column_hints = generate_schema_context(question, schema_info)
    allowed_tables = plan_allowed_tables(question, schema_info, candidate_tables)
    # Try DB-GPT style adapter first (if installed)
    try:
        from dbgpt_integration import available as dbgpt_available, generate_sql_with_dbgpt, validate_and_repair as dbgpt_repair
        if dbgpt_available():
            raw = generate_sql_with_dbgpt(question, schema_context, llm=llm, max_candidates=4)
            _enforce_table_whitelist(raw, allowed_tables)
            cleaned = validate_sql_schema_aware(raw, schema_info)
            return cleaned
    except Exception as e:
        logger.info(f"DBGPT adapter skipped: {e}")

    candidates = generate_sql_candidates(
        question,
        schema_context,
        k=3,
        llm=llm,
        preferred_tables=candidate_tables[:3],
        column_hints=column_hints,
        allowed_tables=allowed_tables,
    )
    for sql in _rank_candidates([repair_sql_common_issues(c) for c in candidates]):
        try:
            _enforce_table_whitelist(sql, allowed_tables)
            cleaned = validate_sql_schema_aware(sql, schema_info)
            return cleaned
        except Exception as e:
            logger.info(f"Candidate rejected: {e}")
            continue
    best = candidates[0] if candidates else "ERROR_NOT_FOUND"
    if best == "ERROR_NOT_FOUND":
        raise ValueError("LLM failed to produce a SELECT statement")
    # Prefer DBGPT repair if available
    fixed = best
    try:
        from dbgpt_integration import available as dbgpt_available, validate_and_repair as dbgpt_repair
        if dbgpt_available():
            fixed = dbgpt_repair(best, schema_context, llm=llm, db_error="initial validation failed", max_retries=2)
        else:
            fixed = repair_sql(best, schema_context, llm=llm, db_error="initial validation failed", max_retries=2)
    except Exception:
        fixed = repair_sql(best, schema_context, llm=llm, db_error="initial validation failed", max_retries=2)
    _enforce_table_whitelist(fixed, allowed_tables)
    cleaned = validate_sql_schema_aware(fixed, schema_info)
    return cleaned

# -----------------------------
# DB execution
# -----------------------------

def run_sql_query(sql_query: str) -> tuple[List[str], List[tuple], int]:
    conn = pyodbc.connect(MSSQL_CONN_STR)
    cursor = conn.cursor()
    cursor.execute(sql_query)
    columns = [c[0] for c in cursor.description]
    rows = cursor.fetchall()
    row_count = len(rows)
    cursor.close()
    conn.close()
    return columns, rows, row_count

# -----------------------------
# Startup init
# -----------------------------

def initialize_components() -> bool:
    global qdrant, embeddings, llm, db_schema
    try:
        embeddings = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL, base_url=OLLAMA_BASE)
        # Qdrant (optional for schema search here; kept for parity with existing project)
        url_parts = QDRANT_URL.replace("http://", "").replace("https://", "").split(":")
        host = url_parts[0]
        port = int(url_parts[1].replace("/", ""))
        from qdrant_client import QdrantClient
        client = QdrantClient(host=host, port=port)
        qdrant = QdrantVectorStore(collection_name=COLLECTION_NAME, embedding=embeddings, client=client)

        llm = ChatOllama(model=OLLAMA_CHAT_MODEL, temperature=0.1, base_url=OLLAMA_BASE)
        logger.info("Discovering database schema...")
        db_schema = discover_database_schema()
        return True
    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        return False

# -----------------------------
# Routes
# -----------------------------
@app.on_event("startup")
async def startup_event():
    logger.info("Starting SISL RAG API (DB-GPT Style)...")
    if not initialize_components():
        logger.error("Initialization incomplete; API may not function fully.")

@app.get("/", response_model=HealthResponse)
async def root():
    return HealthResponse(status="healthy", message="SISL RAG API is running")

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="healthy", message="SISL RAG API is running")

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    try:
        if llm is None or db_schema is None:
            raise RuntimeError("Components not initialized")

        # Simple conversation/DB classification
        intent_prompt = (
            "Classify as 'database_query' or 'general_conversation':\n"
            f"Input: {request.question}\n"
        )
        intent = str(llm.invoke(intent_prompt).content).strip().lower()
        if "general_conversation" in intent:
            text = str(llm.invoke(f"Respond briefly: {request.question}").content).strip()
            return QueryResponse(
                question=request.question,
                generated_sql="",
                results=[{"response": text}],
                columns=["response"],
                row_count=1,
                success=True,
            )

        generated_sql = generate_sql_dbgpt_style(request.question, db_schema)
        # Final validation gate before execution
        final_sql = validate_sql_schema_aware(generated_sql, db_schema)
        columns, rows, row_count = run_sql_query(final_sql)
        results: List[Dict[str, Any]] = []
        for row in rows:
            row_dict: Dict[str, Any] = {}
            for i, value in enumerate(row):
                row_dict[columns[i]] = value.isoformat() if hasattr(value, "isoformat") else (str(value) if value is not None else None)
            results.append(row_dict)
        return QueryResponse(
            question=request.question,
            generated_sql=final_sql,
            results=results,
            columns=columns,
            row_count=row_count,
            success=True,
        )
    except Exception as e:
        logger.error(f"Error: {e}")
        return QueryResponse(
            question=request.question,
            generated_sql="",
            results=[],
            columns=[],
            row_count=0,
            success=False,
            error=str(e),
        )

@app.post("/query/stream")
async def query_stream(request: StreamingQueryRequest) -> StreamingResponse:
    async def generate() -> AsyncGenerator[str, None]:
        try:
            yield f"data: {json.dumps(StreamingChunk(type='status', data={'message': 'Processing query...'}, timestamp=asyncio.get_event_loop().time()).dict())}\n\n"
            if llm is None or db_schema is None:
                raise RuntimeError("Components not initialized")

            # Quick classification
            intent_prompt = (
                "Classify as 'database_query' or 'general_conversation':\n"
                f"Input: {request.question}\n"
            )
            intent = str(llm.invoke(intent_prompt).content).strip().lower()
            yield f"data: {json.dumps(StreamingChunk(type='status', data={'message': f'Intent classified as: {intent}'}, timestamp=asyncio.get_event_loop().time()).dict())}\n\n"

            if "general_conversation" in intent:
                text = str(llm.invoke(f"Respond briefly: {request.question}").content).strip()
                yield f"data: {json.dumps(StreamingChunk(type='results', data={'response': text, 'is_conversation': True}, timestamp=asyncio.get_event_loop().time()).dict())}\n\n"
                yield f"data: {json.dumps(StreamingChunk(type='complete', data={'success': True, 'row_count': 1}, timestamp=asyncio.get_event_loop().time()).dict())}\n\n"
                return

            # Generate + validate before exposing SQL
            yield f"data: {json.dumps(StreamingChunk(type='status', data={'message': 'Generating SQL query...'}, timestamp=asyncio.get_event_loop().time()).dict())}\n\n"
            sql = generate_sql_dbgpt_style(request.question, db_schema)
            yield f"data: {json.dumps(StreamingChunk(type='sql', data={'sql': sql}, timestamp=asyncio.get_event_loop().time()).dict())}\n\n"

            yield f"data: {json.dumps(StreamingChunk(type='status', data={'message': 'Executing query...'}, timestamp=asyncio.get_event_loop().time()).dict())}\n\n"
            columns, rows, row_count = run_sql_query(sql)

            # Stream results progressively
            batch: List[Dict[str, Any]] = []
            for i, row in enumerate(rows):
                row_dict: Dict[str, Any] = {}
                for j, value in enumerate(row):
                    row_dict[columns[j]] = value.isoformat() if hasattr(value, "isoformat") else (str(value) if value is not None else None)
                batch.append(row_dict)
                if (i + 1) % 5 == 0:
                    yield f"data: {json.dumps(StreamingChunk(type='results', data={'rows': batch, 'partial': True}, timestamp=asyncio.get_event_loop().time()).dict())}\n\n"
                    batch = []
            if batch:
                yield f"data: {json.dumps(StreamingChunk(type='results', data={'rows': batch, 'partial': False}, timestamp=asyncio.get_event_loop().time()).dict())}\n\n"
            yield f"data: {json.dumps(StreamingChunk(type='complete', data={'success': True, 'row_count': row_count, 'columns': columns}, timestamp=asyncio.get_event_loop().time()).dict())}\n\n"
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"data: {json.dumps(StreamingChunk(type='error', data={'error': str(e)}, timestamp=asyncio.get_event_loop().time()).dict())}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
        },
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)