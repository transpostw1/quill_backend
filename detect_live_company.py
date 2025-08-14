import argparse
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pyodbc


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("detect_live_company")


PROBE_TABLES: Dict[str, List[str]] = {
    # table_name_suffix: preferred date columns in order
    "Sales Invoice Header": ["Posting Date", "Document Date"],
    "Sales Header": ["Posting Date", "Document Date"],
    "Cust_ Ledger Entry": ["Posting Date"],
    "Item Ledger Entry": ["Posting Date"],
    "Sales Invoice Line": ["Posting Date"],  # may not exist; will fallback
}


def get_companies(conn: pyodbc.Connection) -> List[str]:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT DISTINCT LEFT(TABLE_NAME, CHARINDEX('$', TABLE_NAME) - 1) AS company
        FROM INFORMATION_SCHEMA.TABLES
        WHERE TABLE_TYPE='BASE TABLE' AND CHARINDEX('$', TABLE_NAME) > 0
        ORDER BY company
        """
    )
    rows = cur.fetchall()
    cur.close()
    return [r[0] for r in rows if r[0]]


def table_exists(conn: pyodbc.Connection, schema: str, table: str) -> bool:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT 1
        FROM INFORMATION_SCHEMA.TABLES
        WHERE TABLE_SCHEMA=? AND TABLE_NAME=? AND TABLE_TYPE='BASE TABLE'
        """,
        (schema, table),
    )
    row = cur.fetchone()
    cur.close()
    return row is not None


def get_latest_date_or_rowversion(
    conn: pyodbc.Connection, db_name: str, schema: str, table: str, date_columns: List[str]
) -> Tuple[Optional[datetime], Optional[int]]:
    cur = conn.cursor()
    # First try preferred date columns
    for col in date_columns:
        try:
            sql = f"SELECT MAX([{col}]) FROM [{db_name}].[{schema}].[{table}]"
            cur.execute(sql)
            val = cur.fetchone()[0]
            if val is not None:
                cur.close()
                return (val, None)
        except Exception:
            continue
    # Fallback to rowversion as BIGINT for recency comparison
    try:
        sql = f"SELECT MAX(CAST([timestamp] AS BIGINT)) FROM [{db_name}].[{schema}].[{table}]"
        cur.execute(sql)
        rv = cur.fetchone()[0]
        cur.close()
        return (None, int(rv) if rv is not None else None)
    except Exception:
        cur.close()
        return (None, None)


def get_db_name_from_conn(conn_str: str) -> str:
    parts = [p.strip() for p in conn_str.split(";") if p.strip()]
    for p in parts:
        up = p.upper()
        if up.startswith("DATABASE="):
            return p.split("=", 1)[1].strip()
        if up.startswith("INITIAL CATALOG="):
            return p.split("=", 1)[1].strip()
    return "SISL Live"


def main():
    parser = argparse.ArgumentParser(description="Detect likely live NAV company by recent activity")
    parser.add_argument(
        "--mssql-conn",
        required=True,
        help="ODBC connection string for SQL Server",
    )
    args = parser.parse_args()

    conn = pyodbc.connect(args.mssql_conn)
    db_name = get_db_name_from_conn(args.mssql_conn)
    companies = get_companies(conn)
    if not companies:
        logger.info("No company-prefixed tables found.")
        return

    logger.info("Found companies: %s", ", ".join(companies))

    scores: List[Tuple[str, Optional[datetime], Optional[int], str]] = []

    for company in companies:
        best_date: Optional[datetime] = None
        best_rowversion: Optional[int] = None
        best_src: str = ""
        for suffix, date_cols in PROBE_TABLES.items():
            table = f"{company}${suffix}"
            if not table_exists(conn, "dbo", table):
                continue
            dt, rv = get_latest_date_or_rowversion(conn, db_name, "dbo", table, date_cols)
            if dt is not None:
                if best_date is None or dt > best_date:
                    best_date = dt
                    best_src = f"{table} ({date_cols[0]} or date)"
            elif rv is not None:
                if best_rowversion is None or rv > best_rowversion:
                    best_rowversion = rv
                    best_src = f"{table} ([timestamp] rowversion)"

        scores.append((company, best_date, best_rowversion, best_src))

    # Rank: first by date desc, fallback by rowversion desc
    def _key(item: Tuple[str, Optional[datetime], Optional[int], str]):
        _, dt, rv, _ = item
        return (
            0 if dt is not None else 1,  # prioritize those with a date
            -(dt.timestamp()) if dt is not None else 0,
            -(rv or 0),
        )

    scores.sort(key=_key)

    logger.info("\nLikely live companies by recent activity:")
    for company, dt, rv, src in scores:
        if dt is not None:
            logger.info("  - %s: last date %s via %s", company, dt.strftime("%Y-%m-%d"), src)
        elif rv is not None:
            logger.info("  - %s: last rowversion %s via %s", company, rv, src)
        else:
            logger.info("  - %s: no signal detected", company)

    conn.close()


if __name__ == "__main__":
    main()

