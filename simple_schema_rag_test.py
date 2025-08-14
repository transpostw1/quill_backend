import argparse
import logging
import os
from typing import List, Dict, Tuple, Optional, Set

import pyodbc
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

# LangChain integrations
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("simple_schema_rag_test")


def get_db_name_from_conn(conn_str: str) -> str:
    try:
        parts = [p.strip() for p in conn_str.split(";") if p.strip()]
        for p in parts:
            up = p.upper()
            if up.startswith("DATABASE="):
                return p.split("=", 1)[1].strip()
            if up.startswith("INITIAL CATALOG="):
                return p.split("=", 1)[1].strip()
    except Exception:
        pass
    return "SISL Live"


def mssql_list_tables(conn_str: str) -> List[Tuple[str, str]]:
    connection = pyodbc.connect(conn_str)
    cursor = connection.cursor()
    cursor.execute(
        """
        SELECT TABLE_SCHEMA, TABLE_NAME
        FROM INFORMATION_SCHEMA.TABLES
        WHERE TABLE_TYPE = 'BASE TABLE'
        ORDER BY TABLE_SCHEMA, TABLE_NAME
        """
    )
    rows = cursor.fetchall()
    cursor.close()
    connection.close()
    return [(r[0], r[1]) for r in rows]


def mssql_table_has_rows(conn_str: str, db_name: str, schema: str, table: str) -> Optional[int]:
    try:
        connection = pyodbc.connect(conn_str)
        cursor = connection.cursor()
        # Fully qualify with brackets; table may contain spaces or '$'
        sql = f"SELECT COUNT(*) FROM [{db_name}].[{schema}].[{table}]"
        cursor.execute(sql)
        count = cursor.fetchone()[0]
        cursor.close()
        connection.close()
        return int(count)
    except Exception as e:
        logger.warning(f"Row count check failed for {schema}.{table}: {e}")
        return None


def qdrant_collection_info(client: QdrantClient, collection: str) -> Dict:
    info = client.get_collection(collection)
    # Normalize to dict across pydantic versions
    info_dict: Dict[str, any] = {}
    try:
        if hasattr(info, "dict"):
            info_dict = info.dict()
        elif hasattr(info, "model_dump"):
            info_dict = info.model_dump()
        else:
            info_dict = dict(info.__dict__)
    except Exception:
        info_dict = {}

    result: Dict[str, any] = {
        "status": info_dict.get("status", getattr(info, "status", "unknown")),
    }

    # Vector size (dimension) best-effort
    try:
        cfg = getattr(info, "config", None) or info_dict.get("config")
        params = getattr(cfg, "params", None) if cfg is not None else None
        vectors_cfg = getattr(params, "vectors_config", None) if params is not None else None
        if vectors_cfg is not None and getattr(vectors_cfg, "size", None) is not None:
            result["vector_size"] = vectors_cfg.size
    except Exception:
        pass

    # Points count via count API (reliable across versions)
    try:
        cnt = client.count(collection_name=collection, exact=True)
        result["points_count"] = int(getattr(cnt, "count", 0))
    except Exception:
        result["points_count"] = None

    return result


def qdrant_distinct_tables(client: QdrantClient, collection: str, scan_limit: int = 10000) -> Set[str]:
    """Scroll through points and infer distinct table names from payload.

    We look for payload keys 'table' or 'exact_table'. If 'type'=='table' then the payload itself
    is a table entry. For column entries, we still collect payload['table'] if present.
    """
    offset = None
    distinct_tables: Set[str] = set()
    scanned = 0
    while True:
        if scanned >= scan_limit:
            break
        limit = min(1024, scan_limit - scanned)
        points, offset = client.scroll(
            collection_name=collection,
            scroll_filter=None,
            limit=limit,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not points:
            break
        scanned += len(points)
        for p in points:
            payload = p.payload or {}
            t = payload.get("exact_table") or payload.get("table")
            if isinstance(t, str) and t:
                distinct_tables.add(t)
        if offset is None:
            break
    return distinct_tables


def qdrant_count_type_table(client: QdrantClient, collection: str) -> int:
    """Count points where payload.type == 'table'."""
    try:
        filt = qmodels.Filter(must=[qmodels.FieldCondition(key="type", match=qmodels.MatchValue(value="table"))])
        res = client.count(collection_name=collection, count_filter=filt, exact=True)
        return int(getattr(res, "count", 0))
    except Exception as e:
        logger.warning(f"Count(type=table) failed: {e}")
        return 0


def run_schema_rag(
    qdrant_url: str,
    collection: str,
    ollama_base: str,
    question: str,
    top_k: int = 10,
):
    logger.info("Connecting to Qdrant and performing similarity search...")
    client = QdrantClient(host=qdrant_url.replace("http://", "").replace("https://", "").split(":")[0],
                          port=int(qdrant_url.split(":")[-1].strip("/")))
    embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=ollama_base)
    vs = QdrantVectorStore(collection_name=collection, embedding=embeddings, client=client)
    docs = vs.similarity_search(question, k=top_k)
    logger.info(f"Top-{top_k} retrieved docs for question: {question}")
    for i, d in enumerate(docs, 1):
        logger.info(f"Doc {i} content (truncated 300 chars): {d.page_content[:300]!r}")
        if d.metadata:
            logger.info(f"Doc {i} metadata: {d.metadata}")


def main():
    p = argparse.ArgumentParser(description="Simple schema RAG + consistency debug (MSSQL vs Qdrant)")
    p.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL", "http://127.0.0.1:6333"))
    p.add_argument("--qdrant-collection", default=os.getenv("QDRANT_COLLECTION", "sisl_uat_nav_768"))
    p.add_argument("--ollama-base", default=os.getenv("OLLAMA_BASE", "http://127.0.0.1:11434"))
    p.add_argument("--mssql-conn", default=os.getenv("MSSQL_CONN_STR", \
        "DRIVER={ODBC Driver 17 for SQL Server};SERVER=192.168.1.44,1433;DATABASE=SISL Live;UID=test;PWD=Test@345;TrustServerCertificate=yes;"))
    p.add_argument("--question", default="How many customers do we have?")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--scan-limit", type=int, default=10000, help="Max points to scan from Qdrant for distinct tables")
    p.add_argument("--test-customer-count", action="store_true", help="Attempt COUNT(*) on best-guess Customer table")
    args = p.parse_args()

    logger.info("=== CONFIG ===")
    logger.info(f"Qdrant: {args.qdrant_url} | Collection: {args.qdrant_collection}")
    logger.info(f"Ollama:  {args.ollama_base}")
    db_name = get_db_name_from_conn(args.mssql_conn)
    logger.info(f"MSSQL DB: {db_name}")

    # 1) MSSQL tables
    logger.info("\n=== MSSQL: Listing tables ===")
    mssql_tables = mssql_list_tables(args.mssql_conn)
    logger.info(f"Total MSSQL base tables: {len(mssql_tables)}")
    logger.info(f"First 15 tables: {mssql_tables[:15]}")
    mssql_customer_like = [(s, t) for s, t in mssql_tables if "customer" in t.lower()]
    logger.info(f"Customer-like tables (MSSQL): {[f'{s}.{t}' for s,t in mssql_customer_like[:10]]}")

    # 2) Qdrant stats
    logger.info("\n=== Qdrant: Collection info ===")
    qclient = QdrantClient(host=args.qdrant_url.replace("http://", "").replace("https://", "").split(":")[0],
                           port=int(args.qdrant_url.split(":")[-1].strip("/")))
    info = qdrant_collection_info(qclient, args.qdrant_collection)
    logger.info(f"Collection info: {info}")
    total_points = info.get("points_count")
    type_table_count = qdrant_count_type_table(qclient, args.qdrant_collection)
    logger.info(f"Total points: {total_points} | Points(type=table): {type_table_count}")

    distinct_tables = qdrant_distinct_tables(qclient, args.qdrant_collection, scan_limit=args.scan_limit)
    logger.info(f"Distinct table names inferred from payload (scanned <= {args.scan_limit}): {len(distinct_tables)}")
    sample_tables = list(distinct_tables)[:15]
    logger.info(f"Sample Qdrant tables: {sample_tables}")
    qdrant_customer_like = [t for t in distinct_tables if "customer" in t.lower()]
    logger.info(f"Customer-like tables (Qdrant): {qdrant_customer_like[:10]}")

    # 3) Compare sets (rough)
    mssql_table_names_only = set(t for _, t in mssql_tables)
    qdrant_table_names_only = set(distinct_tables)
    missing_in_qdrant = sorted(list(mssql_table_names_only - qdrant_table_names_only))[:30]
    extra_in_qdrant = sorted(list(qdrant_table_names_only - mssql_table_names_only))[:30]
    logger.info(f"\nTables in MSSQL but NOT in Qdrant (sample up to 30): {missing_in_qdrant}")
    logger.info(f"Tables in Qdrant but NOT in MSSQL (sample up to 30): {extra_in_qdrant}")

    # 4) Simple RAG retrieval on schema
    logger.info("\n=== RAG retrieval for question ===")
    run_schema_rag(args.qdrant_url, args.qdrant_collection, args.ollama_base, args.question, top_k=args.top_k)

    # 5) Optional: sanity COUNT on candidate customer table
    if args.test_customer_count and mssql_customer_like:
        best = None
        # Prefer tables ending with $Customer, else plain Customer
        for _, t in mssql_customer_like:
            if t.lower().endswith("$customer"):
                best = t
                break
        if best is None:
            for _, t in mssql_customer_like:
                if t.lower() == "customer" or t.lower().endswith(" customer"):
                    best = t
                    break
        if best is None:
            best = mssql_customer_like[0][1]
        logger.info(f"Attempting COUNT(*) on candidate Customer table: dbo.{best}")
        count = mssql_table_has_rows(args.mssql_conn, db_name, "dbo", best)
        logger.info(f"COUNT result: {count}")

    logger.info("\n=== Done ===")


if __name__ == "__main__":
    main()

