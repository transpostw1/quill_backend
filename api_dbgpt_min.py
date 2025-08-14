from __future__ import annotations

# Minimal DB-GPT-only SQL RAG API (no extra heuristics)
# - Requires: dbgpt (PyPI) and sqlglot
# - Endpoints: /health, /query, /query/stream
# - Uses DB-GPT adapter exclusively for SQL generation/repair

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, AsyncGenerator

import asyncio
import logging
import json
import pyodbc
import re

# Dependency checks
try:
    import dbgpt  # type: ignore  # noqa: F401
    _DBGPT_OK = True
except Exception:
    _DBGPT_OK = False

try:
    from sqlglot import parse_one  # type: ignore  # noqa: F401
    _SQLGLOT_OK = True
except Exception:
    _SQLGLOT_OK = False

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain.prompts import ChatPromptTemplate

# Adapter wrapper
from dbgpt_integration import available as dbgpt_available, generate_sql_with_dbgpt, validate_and_repair

# -----------------------------
# Config (keep minimal)
# -----------------------------
MSSQL_CONN_STR = (
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=192.168.1.44,1433;"
    "DATABASE=SISL Live;"
    "UID=test;"
    "PWD=Test@345;"
    "TrustServerCertificate=yes;"
)

OLLAMA_BASE = "http://192.168.0.120:11434"
OLLAMA_CHAT_MODEL = "mistral:latest"
OLLAMA_EMBED_MODEL = "nomic-embed-text"

# Qdrant schema collection (existing)
QDRANT_URL = "http://192.168.0.120:6333/"
COLLECTION_NAME = "sisl_uat_nav_768"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api_dbgpt_min")

app = FastAPI(title="SISL RAG API (DB-GPT Minimal)", version="1.0.0")
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
llm: ChatOllama | None = None
embeddings: OllamaEmbeddings | None = None
qdrant: QdrantVectorStore | None = None

# -----------------------------
# Utilities
# -----------------------------

def run_sql(sql_query: str) -> tuple[List[str], List[tuple], int]:
    conn = pyodbc.connect(MSSQL_CONN_STR)
    cur = conn.cursor()
    cur.execute(sql_query)
    cols = [c[0] for c in cur.description]
    rows = cur.fetchall()
    n = len(rows)
    cur.close()
    conn.close()
    return cols, rows, n


def minimal_schema_context() -> str:
    # Keep context minimal; DB-GPT should handle candidate generation/repair.
    return "Use SQL Server T-SQL. Qualify tables as [SISL Live].[dbo].[Table]."


def qdrant_schema_context(question: str) -> str:
    global qdrant
    try:
        if qdrant is None:
            return minimal_schema_context()
        hits = qdrant.similarity_search(question, k=20)
        if not hits:
            return minimal_schema_context()
        # Build focused context from retrieved docs
        lines = ["SCHEMA CONTEXT (RETRIEVED):"]
        all_text = []
        for d in hits:
            content = d.page_content or ""
            lines.append(content[:500])
            all_text.append(content)
        # Extract allowed tables from retrieved content
        joined = "\n".join(all_text)
        tbls = re.findall(r"\[SISL Live\]\.\[dbo\]\.\[([^\]]+)\]", joined)
        allowed = []
        seen = set()
        for t in tbls:
            if t not in seen:
                allowed.append(t)
                seen.add(t)
        if allowed:
            lines.append("\nALLOWED TABLES (use only these):")
            for t in allowed[:30]:
                lines.append(f"- [SISL Live].[dbo].[{t}]")
        # Add strict instruction to avoid markdown/prose
        lines.append("\nINSTRUCTION: Return ONLY one T-SQL statement. No markdown fences. No prose. No explanations.")
        # Optionally include allowed columns for top tables to anchor column names
        try:
            conn = pyodbc.connect(MSSQL_CONN_STR)
            top_tables = allowed[:3]
            if top_tables:
                lines.append("\nALLOWED COLUMNS (use only columns from these lists):")
                for t in top_tables:
                    cols = _get_table_columns(conn, t)
                    if cols:
                        preview = ", ".join(f"[{c}]" for c in cols[:25])
                        lines.append(f"- [SISL Live].[dbo].[{t}] : {preview}")
            conn.close()
        except Exception:
            pass
        return "\n".join(lines)
    except Exception:
        return minimal_schema_context()

# -----------------------------
# Startup
# -----------------------------
@app.on_event("startup")
async def startup_event():
    global llm
    global embeddings, qdrant
    if not _DBGPT_OK:
        logger.error("DB-GPT package not installed. Install with: pip install dbgpt sqlglot")
    if not _SQLGLOT_OK:
        logger.error("sqlglot not installed. Install with: pip install sqlglot")
    llm = ChatOllama(model=OLLAMA_CHAT_MODEL, temperature=0.1, base_url=OLLAMA_BASE)
    # Initialize Qdrant vector store against existing schema_vectors
    try:
        embeddings = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL, base_url=OLLAMA_BASE)
        # Parse Qdrant URL
        parts = QDRANT_URL.replace("http://", "").replace("https://", "").split(":")
        host = parts[0]
        port = int(parts[1].replace("/", "")) if len(parts) > 1 else 6333
        from qdrant_client import QdrantClient
        client = QdrantClient(host=host, port=port)
        qdrant = QdrantVectorStore(collection_name=COLLECTION_NAME, embedding=embeddings, client=client)
        logger.info("Qdrant schema_vectors ready for retrieval")
    except Exception as e:
        logger.info(f"Qdrant retrieval disabled: {e}")
    logger.info("DB-GPT Minimal API started")


# -----------------------------
# SQL sanitization helpers
# -----------------------------
def _strip_code_fences(text: str) -> str:
    t = text.strip()
    t = t.replace("```sql", "").replace("```", "").strip()
    # Remove common leading labels
    for lead in ("SQL:", "T-SQL:", "TSql:", "Query:"):
        if t.startswith(lead):
            t = t[len(lead):].strip()
    return t


def sanitize_sql(text: str) -> str:
    """Extract the first SELECT statement; remove markdown/prose."""
    t = _strip_code_fences(text)
    m = re.search(r"(?is)\bselect\b.*?;", t)
    if m:
        return " ".join(m.group(0).split())
    m2 = re.search(r"(?is)\bselect\b.*", t)
    if m2:
        cand = " ".join(m2.group(0).split())
        if not cand.endswith(";"):
            cand += ";"
        return cand
    return t


# -----------------------------
# Runtime column validation (DB lookup)
# -----------------------------
def _map_aliases(sql: str) -> Dict[str, str]:
    """Extract alias -> table name from FROM/JOIN clauses.
    Returns lowercased aliases mapped to table names as they appear in brackets.
    """
    alias_map: Dict[str, str] = {}
    # FROM [DB].[dbo].[Table] [alias] or AS alias
    for m in re.finditer(r"FROM\s+\[SISL Live\]\.\[dbo\]\.\[([^\]]+)\](?:\s+(?:AS\s+)?(\w+))?", sql, re.IGNORECASE):
        table, alias = m.group(1), m.group(2)
        if alias:
            alias_map[alias.lower()] = table
    # JOIN [DB].[dbo].[Table] [alias]
    for m in re.finditer(r"JOIN\s+\[SISL Live\]\.\[dbo\]\.\[([^\]]+)\](?:\s+(?:AS\s+)?(\w+))?", sql, re.IGNORECASE):
        table, alias = m.group(1), m.group(2)
        if alias:
            alias_map[alias.lower()] = table
    return alias_map


def _extract_alias_col_refs(sql: str) -> List[tuple[str, str]]:
    """Find alias.[Column] references in the SQL."""
    return [(a, c) for a, c in re.findall(r"(\w+)\.\[([^\]]+)\]", sql)]


def _get_table_columns(conn: pyodbc.Connection, table: str) -> List[str]:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_NAME = ? ORDER BY ORDINAL_POSITION
        """,
        table,
    )
    cols = [r[0] for r in cur.fetchall()]
    cur.close()
    return cols


def validate_columns_runtime(sql: str) -> None:
    """Validate that alias.[Column] references exist in the referenced tables.
    Raises ValueError with details if invalid columns are found.
    """
    alias_map = _map_aliases(sql)
    refs = _extract_alias_col_refs(sql)
    # Also capture fully-qualified refs: [DB].[dbo].[Table].[Column]
    fq_refs = re.findall(r"\[SISL Live\]\.?\[dbo\]\.?\[([^\]]+)\]\.\[([^\]]+)\]", sql, re.IGNORECASE)
    if not refs and not fq_refs:
        return
    conn = pyodbc.connect(MSSQL_CONN_STR)
    try:
        cache: Dict[str, List[str]] = {}
        errors: List[str] = []
        # alias-qualified
        for alias, col in refs:
            table = alias_map.get(alias.lower())
            if table:
                if table not in cache:
                    cache[table] = _get_table_columns(conn, table)
                if col not in cache[table]:
                    errors.append(f"{alias}.[{col}] not found in [{table}]")
        # fully-qualified
        for table, col in fq_refs:
            if table not in cache:
                cache[table] = _get_table_columns(conn, table)
            if col not in cache[table]:
                errors.append(f"[dbo].[{table}].[{col}] not found")
        if errors:
            raise ValueError("; ".join(errors))
    finally:
        conn.close()

# -----------------------------
# Routes
# -----------------------------
@app.get("/", response_model=HealthResponse)
async def root():
    return HealthResponse(status="healthy", message="DB-GPT Minimal API is running")

@app.get("/health", response_model=HealthResponse)
async def health():
    if not _DBGPT_OK:
        return HealthResponse(status="degraded", message="Install dbgpt to enable SQL generation")
    return HealthResponse(status="healthy", message="OK")

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    try:
        if not _DBGPT_OK or not dbgpt_available():
            raise RuntimeError("DB-GPT not available. Install with: pip install dbgpt sqlglot")
        if llm is None:
            raise RuntimeError("LLM not initialized")

        schema_ctx = qdrant_schema_context(request.question)
        # Generate via DB-GPT path
        raw_sql = generate_sql_with_dbgpt(request.question, schema_ctx, llm=llm, max_candidates=4)
        raw_sql = sanitize_sql(raw_sql)
        # Attempt repair if needed
        sql_fixed = validate_and_repair(raw_sql, schema_ctx, llm=llm, db_error=None, max_retries=2)
        sql_fixed = sanitize_sql(sql_fixed)
        # Execute (with one-shot repair on DB error)
        try:
            cols, rows, n = run_sql(sql_fixed)
        except Exception as db_err:
            # Feed DB error back to DB-GPT for a single repair pass
            repaired = validate_and_repair(sql_fixed, schema_ctx, llm=llm, db_error=str(db_err), max_retries=1)
            repaired = sanitize_sql(repaired)
            cols, rows, n = run_sql(repaired)
            sql_fixed = repaired
        # Format results
        out: List[Dict[str, Any]] = []
        for row in rows:
            d: Dict[str, Any] = {}
            for i, v in enumerate(row):
                d[cols[i]] = v.isoformat() if hasattr(v, "isoformat") else (str(v) if v is not None else None)
            out.append(d)
        return QueryResponse(
            question=request.question,
            generated_sql=sql_fixed,
            results=out,
            columns=cols,
            row_count=n,
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
    async def gen() -> AsyncGenerator[str, None]:
        try:
            yield f"data: {json.dumps(StreamingChunk(type='status', data={'message': 'Processing...'}, timestamp=asyncio.get_event_loop().time()).dict())}\n\n"
            if not _DBGPT_OK or not dbgpt_available():
                raise RuntimeError("DB-GPT not available. Install with: pip install dbgpt sqlglot")
            if llm is None:
                raise RuntimeError("LLM not initialized")

            schema_ctx = qdrant_schema_context(request.question)
            yield f"data: {json.dumps(StreamingChunk(type='status', data={'message': 'Generating SQL (DB-GPT)...'}, timestamp=asyncio.get_event_loop().time()).dict())}\n\n"
            raw_sql = generate_sql_with_dbgpt(request.question, schema_ctx, llm=llm, max_candidates=4)
            raw_sql = sanitize_sql(raw_sql)
            fixed = validate_and_repair(raw_sql, schema_ctx, llm=llm, db_error=None, max_retries=2)
            fixed = sanitize_sql(fixed)
            yield f"data: {json.dumps(StreamingChunk(type='sql', data={'sql': fixed}, timestamp=asyncio.get_event_loop().time()).dict())}\n\n"

            yield f"data: {json.dumps(StreamingChunk(type='status', data={'message': 'Executing...'}, timestamp=asyncio.get_event_loop().time()).dict())}\n\n"
            try:
                cols, rows, n = run_sql(fixed)
            except Exception as db_err:
                # Attempt one repair on DB error
                yield f"data: {json.dumps(StreamingChunk(type='status', data={'message': 'Repairing SQL with DB error feedback...'}, timestamp=asyncio.get_event_loop().time()).dict())}\n\n"
                repaired = validate_and_repair(fixed, schema_ctx, llm=llm, db_error=str(db_err), max_retries=1)
                repaired = sanitize_sql(repaired)
                yield f"data: {json.dumps(StreamingChunk(type='sql', data={'sql': repaired}, timestamp=asyncio.get_event_loop().time()).dict())}\n\n"
                cols, rows, n = run_sql(repaired)
            batch: List[Dict[str, Any]] = []
            for i, row in enumerate(rows):
                d: Dict[str, Any] = {}
                for j, v in enumerate(row):
                    d[cols[j]] = v.isoformat() if hasattr(v, "isoformat") else (str(v) if v is not None else None)
                batch.append(d)
                if (i + 1) % 5 == 0:
                    yield f"data: {json.dumps(StreamingChunk(type='results', data={'rows': batch, 'partial': True}, timestamp=asyncio.get_event_loop().time()).dict())}\n\n"
                    batch = []
            if batch:
                yield f"data: {json.dumps(StreamingChunk(type='results', data={'rows': batch, 'partial': False}, timestamp=asyncio.get_event_loop().time()).dict())}\n\n"
            yield f"data: {json.dumps(StreamingChunk(type='complete', data={'success': True, 'row_count': n, 'columns': cols}, timestamp=asyncio.get_event_loop().time()).dict())}\n\n"
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"data: {json.dumps(StreamingChunk(type='error', data={'error': str(e)}, timestamp=asyncio.get_event_loop().time()).dict())}\n\n"

    return StreamingResponse(
        gen(),
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