#!/usr/bin/env python3
"""
Re-embed NAV (Dynamics NAV/Business Central) schema into Qdrant using Ollama (768-d)

What it does
- Connects to MSSQL and discovers business tables and columns
- Builds table and column documents with NAV-style bracketed names
- Embeds with Ollama 'nomic-embed-text' (768-d) and upserts to Qdrant

Usage (run on server)
  source /var/www/quill_backend/venv/bin/activate
  python reembed_nav_schema.py \
    --collection-name sisl_uat_nav_768 \
    --qdrant-url http://134.122.23.222:6333 \
    --ollama-base http://192.168.0.120:11434 \
    --db-conn "DRIVER={ODBC Driver 17 for SQL Server};SERVER=192.168.1.44,1433;DATABASE=SISL Live;UID=test;PWD=Test@345;TrustServerCertificate=yes;" \
    --batch-size 200 \
    --drop-if-exists \
    --company-prefix "SISL$"

Then point your API to the new collection name.
"""
from __future__ import annotations

import argparse
import sys
import time
import logging
from typing import Dict, List, Any

import pyodbc

from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore

# Optional: create collection explicitly
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("reembed_nav_schema")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Re-embed NAV schema into Qdrant with Ollama (768-d)")
    p.add_argument("--collection-name", required=True, help="Target Qdrant collection name to create/write")
    p.add_argument("--qdrant-url", required=True, help="Qdrant base URL, e.g., http://134.122.23.222:6333")
    p.add_argument("--ollama-base", required=True, help="Ollama base URL, e.g., http://192.168.0.120:11434")
    p.add_argument("--db-conn", required=True, help="MSSQL ODBC connection string")
    p.add_argument("--batch-size", type=int, default=200, help="Upsert batch size")
    p.add_argument("--drop-if-exists", action="store_true", help="Drop the collection if it already exists")
    p.add_argument("--company-prefix", default=None, help="Only include tables whose names start with this prefix, e.g., 'SISL$'")
    p.add_argument(
        "--extra-exclude",
        default="buffer,report,layout,temp,history,archive,queue,workflow,notification",
        help="Comma-separated substrings; tables containing any will be excluded",
    )
    return p.parse_args()


def is_excluded_table(name: str) -> bool:
    n = name.lower().strip()
    exclude = [
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
    return any(x in n for x in exclude)


def is_config_table(name: str) -> bool:
    n = name.lower().strip()
    cfg = ["setup", "code", "chart", "config", "dimension", "template", "lookup"]
    return any(x in n for x in cfg)


def is_noise_table(name: str, extra_substrings: list[str]) -> bool:
    n = name.lower().strip()
    return any(x for x in extra_substrings if x and x in n)


def discover_schema(conn_str: str, company_prefix: str | None = None, extra_exclude: list[str] | None = None) -> Dict[str, Any]:
    logger.info("Connecting to MSSQL for schema discovery...")
    conn = pyodbc.connect(conn_str)
    cur = conn.cursor()

    schema: Dict[str, Any] = {"tables": {}, "relationships": []}

    cur.execute(
        """
        SELECT TABLE_SCHEMA, TABLE_NAME
        FROM INFORMATION_SCHEMA.TABLES
        WHERE TABLE_TYPE='BASE TABLE'
        ORDER BY TABLE_NAME
        """
    )
    rows = cur.fetchall()
    logger.info("Found %d tables (pre-filter)", len(rows))

    extra_exclude = extra_exclude or []
    for schema_name, table_name in rows:
        if is_excluded_table(table_name):
            continue
        if company_prefix:
            if not table_name.lower().startswith(company_prefix.lower()):
                continue
        if is_config_table(table_name):
            continue
        if is_noise_table(table_name, extra_exclude):
            continue
        schema["tables"][table_name] = {"schema": schema_name, "columns": []}
        cur.execute(
            """
            SELECT COLUMN_NAME, DATA_TYPE
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_NAME = ?
            ORDER BY ORDINAL_POSITION
            """,
            table_name,
        )
        cols = cur.fetchall()
        schema["tables"][table_name]["columns"] = [{"name": c, "type": t} for c, t in cols]

    # Relationships (FKs) optional, not required for embeddings
    try:
        cur.execute(
            """
            SELECT tp.name, tr.name
            FROM sys.foreign_keys fk
            INNER JOIN sys.foreign_key_columns fkc ON fk.object_id = fkc.constraint_object_id
            INNER JOIN sys.tables tp ON fkc.parent_object_id = tp.object_id
            INNER JOIN sys.tables tr ON fkc.referenced_object_id = tr.object_id
            """
        )
        for parent, ref in cur.fetchall():
            schema["relationships"].append({"source_table": parent, "target_table": ref})
    except Exception as e:
        logger.warning("FK discovery skipped: %s", e)

    cur.close()
    conn.close()

    kept = list(schema["tables"].keys())
    logger.info("Kept %d tables after filters (business focus)", len(kept))
    return schema


def build_documents(schema: Dict[str, Any], company_db: str = "SISL Live") -> List[Document]:
    docs: List[Document] = []
    for table, meta in schema["tables"].items():
        cols = meta.get("columns", [])
        col_list = ", ".join(f"[{c['name']}] ({c['type']})" for c in cols[:100])
        content = (
            f"TABLE [{company_db}].[dbo].[{table}]\n"
            f"COLUMNS: {col_list}\n"
        )
        docs.append(
            Document(
                page_content=content,
                metadata={
                    "type": "table",
                    "table_name": table,
                    "schema": meta.get("schema", "dbo"),
                },
            )
        )
        for c in cols:
            content_c = (
                f"COLUMN [{company_db}].[dbo].[{table}].[{c['name']}]\nTYPE: {c['type']}"
            )
            docs.append(
                Document(
                    page_content=content_c,
                    metadata={
                        "type": "column",
                        "table_name": table,
                        "column_name": c["name"],
                        "data_type": c["type"],
                    },
                )
            )
    return docs


def ensure_collection(qdrant_url: str, collection_name: str, drop_if_exists: bool) -> QdrantClient:
    # Parse URL
    base = qdrant_url.replace("http://", "").replace("https://", "")
    if ":" in base:
        host, port = base.split(":", 1)
        port = int(port.rstrip("/"))
    else:
        host, port = base, 6333
    client = QdrantClient(host=host, port=port)

    existing = None
    try:
        existing = client.get_collection(collection_name)
    except Exception:
        existing = None

    if existing and drop_if_exists:
        logger.info("Dropping existing collection '%s'...", collection_name)
        client.delete_collection(collection_name)
        existing = None

    if not existing:
        logger.info("Creating collection '%s' (768-d, Cosine)...", collection_name)
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )
    else:
        logger.info("Collection '%s' exists; will upsert", collection_name)
    return client


def upsert_documents(
    docs: List[Document],
    qdrant_client: QdrantClient,
    collection_name: str,
    ollama_base: str,
    batch_size: int,
) -> int:
    embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=ollama_base)
    store = QdrantVectorStore(collection_name=collection_name, embedding=embeddings, client=qdrant_client)

    total = 0
    t0 = time.time()
    for i in range(0, len(docs), batch_size):
        batch = docs[i : i + batch_size]
        store.add_documents(batch)
        total += len(batch)
        logger.info("Upserted %d/%d", total, len(docs))
    logger.info("Upsert complete in %.2fs", time.time() - t0)
    return total


def main() -> int:
    args = parse_args()

    extra_exclude = [s.strip().lower() for s in (args.extra_exclude or "").split(",") if s.strip()]
    schema = discover_schema(args.db_conn, company_prefix=args.company_prefix, extra_exclude=extra_exclude)
    docs = build_documents(schema)

    if not docs:
        logger.error("No documents to index; aborting")
        return 1

    # Preview first few
    for d in docs[:3]:
        logger.info("DOC SAMPLE: %s | %s", d.metadata, (d.page_content[:120] + "...") if len(d.page_content) > 120 else d.page_content)

    client = ensure_collection(args.qdrant_url, args.collection_name, args.drop_if_exists)
    total = upsert_documents(docs, client, args.collection_name, args.ollama_base, args.batch_size)

    logger.info("\nSUMMARY:")
    logger.info("  Collection: %s", args.collection_name)
    logger.info("  Tables indexed: %d", len(schema["tables"]))
    logger.info("  Docs upserted: %d (tables + columns)", total)
    logger.info("  Vector size: 768 (Ollama nomic-embed-text)")

    return 0


if __name__ == "__main__":
    sys.exit(main())