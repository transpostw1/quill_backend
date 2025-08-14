import argparse
import logging
from collections import defaultdict, Counter
from typing import Dict, List, Tuple

import pyodbc


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("export_mssql_schema_txt")


def fetch_tables(cursor, include_views: bool = False) -> List[Tuple[str, str, str]]:
    types = ("'BASE TABLE'", "'VIEW'") if include_views else ("'BASE TABLE'",)
    cursor.execute(
        f"""
        SELECT TABLE_SCHEMA, TABLE_NAME, TABLE_TYPE
        FROM INFORMATION_SCHEMA.TABLES
        WHERE TABLE_TYPE IN ({', '.join(types)})
        ORDER BY TABLE_SCHEMA, TABLE_NAME
        """
    )
    return [(r[0], r[1], r[2]) for r in cursor.fetchall()]


def fetch_columns(cursor) -> List[Tuple[str, str, str, str, int, int, int, str, int]]:
    cursor.execute(
        """
        SELECT 
            TABLE_SCHEMA,
            TABLE_NAME,
            COLUMN_NAME,
            DATA_TYPE,
            ISNULL(CHARACTER_MAXIMUM_LENGTH, 0) AS CHAR_MAX_LEN,
            ISNULL(NUMERIC_PRECISION, 0) AS NUM_PREC,
            ISNULL(NUMERIC_SCALE, 0) AS NUM_SCALE,
            IS_NULLABLE,
            ORDINAL_POSITION
        FROM INFORMATION_SCHEMA.COLUMNS
        ORDER BY TABLE_SCHEMA, TABLE_NAME, ORDINAL_POSITION
        """
    )
    return [
        (
            r[0],
            r[1],
            r[2],
            r[3],
            int(r[4]) if r[4] is not None else 0,
            int(r[5]) if r[5] is not None else 0,
            int(r[6]) if r[6] is not None else 0,
            r[7],
            int(r[8]),
        )
        for r in cursor.fetchall()
    ]


def fetch_primary_keys(cursor) -> Dict[Tuple[str, str], List[Tuple[int, str]]]:
    cursor.execute(
        """
        SELECT 
            tc.TABLE_SCHEMA,
            tc.TABLE_NAME,
            kcu.COLUMN_NAME,
            kcu.ORDINAL_POSITION
        FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS AS tc
        JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE AS kcu
          ON tc.CONSTRAINT_NAME = kcu.CONSTRAINT_NAME
         AND tc.TABLE_SCHEMA = kcu.TABLE_SCHEMA
        WHERE tc.CONSTRAINT_TYPE = 'PRIMARY KEY'
        ORDER BY tc.TABLE_SCHEMA, tc.TABLE_NAME, kcu.ORDINAL_POSITION
        """
    )
    pks: Dict[Tuple[str, str], List[Tuple[int, str]]] = defaultdict(list)
    for row in cursor.fetchall():
        pks[(row[0], row[1])].append((int(row[3]), row[2]))
    return pks


def fetch_foreign_keys(cursor) -> List[Tuple[str, str, str, str, str, str, str]]:
    # Note: This is a simplified FK mapping. For composite keys, entries repeat with increasing positions.
    cursor.execute(
        """
        SELECT 
            fk.TABLE_SCHEMA AS FK_SCHEMA,
            fk.TABLE_NAME  AS FK_TABLE,
            fk.COLUMN_NAME AS FK_COLUMN,
            pk.TABLE_SCHEMA AS PK_SCHEMA,
            pk.TABLE_NAME  AS PK_TABLE,
            pk.COLUMN_NAME AS PK_COLUMN,
            rc.CONSTRAINT_NAME
        FROM INFORMATION_SCHEMA.REFERENTIAL_CONSTRAINTS rc
        JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE fk 
          ON rc.CONSTRAINT_NAME = fk.CONSTRAINT_NAME
         AND rc.CONSTRAINT_SCHEMA = fk.CONSTRAINT_SCHEMA
        JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE pk 
          ON rc.UNIQUE_CONSTRAINT_NAME = pk.CONSTRAINT_NAME
         AND rc.CONSTRAINT_SCHEMA = pk.CONSTRAINT_SCHEMA
        ORDER BY FK_SCHEMA, FK_TABLE, rc.CONSTRAINT_NAME
        """
    )
    return [(r[0], r[1], r[2], r[3], r[4], r[5], r[6]) for r in cursor.fetchall()]


def detect_company_prefixes(tables: List[Tuple[str, str, str]]) -> Counter:
    c = Counter()
    for schema, name, _ in tables:
        if "$" in name:
            prefix = name.split("$", 1)[0]
            c[prefix] += 1
    return c


def export_schema_txt(
    conn_str: str,
    out_path: str,
    include_views: bool = False,
) -> None:
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()

    tables = fetch_tables(cursor, include_views=include_views)
    columns = fetch_columns(cursor)
    pks = fetch_primary_keys(cursor)
    fks = fetch_foreign_keys(cursor)

    # Organize columns by (schema, table)
    cols_by_table: Dict[Tuple[str, str], List[Tuple[int, str, str, int, int, int, str]]] = defaultdict(list)
    for schema, table, col, dtype, char_len, num_prec, num_scale, is_nullable, ord_pos in columns:
        cols_by_table[(schema, table)].append((ord_pos, col, dtype, char_len, num_prec, num_scale, is_nullable))
    for key in cols_by_table:
        cols_by_table[key].sort(key=lambda x: x[0])

    # Group FKs by FK table
    fks_by_table: Dict[Tuple[str, str], List[Tuple[str, str, str, str, str]]] = defaultdict(list)
    for fk_schema, fk_table, fk_col, pk_schema, pk_table, pk_col, cname in fks:
        fks_by_table[(fk_schema, fk_table)].append((cname, fk_col, pk_schema, pk_table, pk_col))

    prefixes = detect_company_prefixes(tables)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("SCHEMA EXPORT (MSSQL)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total tables: {len(tables)}\n")
        f.write(f"Include views: {include_views}\n\n")

        if prefixes:
            f.write("Detected company prefixes (before '$'):\n")
            for pref, cnt in prefixes.most_common():
                f.write(f"  - {pref}: {cnt} tables\n")
            f.write("\n")

        # Tables section
        f.write("TABLES\n")
        f.write("-" * 80 + "\n")
        for schema, name, ttype in tables:
            f.write(f"[{schema}].[{name}] ({ttype})\n")
        f.write("\n")

        # Detailed section per table
        f.write("DETAILS\n")
        f.write("-" * 80 + "\n")
        for schema, name, ttype in tables:
            f.write(f"Table: [{schema}].[{name}]\n")
            # PKs
            pk_cols = pks.get((schema, name), [])
            if pk_cols:
                ordered = [col for _, col in sorted(pk_cols, key=lambda x: x[0])]
                f.write(f"  Primary Key: ({', '.join(ordered)})\n")
            else:
                f.write("  Primary Key: (none)\n")

            # Columns
            f.write("  Columns:\n")
            for ord_pos, col, dtype, char_len, num_prec, num_scale, is_nullable in cols_by_table.get((schema, name), []):
                size = ""
                if dtype.lower() in {"varchar", "nvarchar", "char", "nchar", "varbinary"} and char_len:
                    size = f"({char_len})"
                elif dtype.lower() in {"decimal", "numeric"}:
                    size = f"({num_prec},{num_scale})"
                f.write(f"    - {col}: {dtype}{size} NULLABLE={is_nullable}\n")

            # FKs
            fk_list = fks_by_table.get((schema, name), [])
            if fk_list:
                f.write("  Foreign Keys:\n")
                for cname, fk_col, pk_schema, pk_table, pk_col in fk_list:
                    f.write(f"    - {cname}: {fk_col} -> [{pk_schema}].[{pk_table}].[{pk_col}]\n")
            f.write("\n")

    cursor.close()
    conn.close()
    logger.info("Export complete: %s", out_path)


def main():
    parser = argparse.ArgumentParser(description="Export MSSQL schema to a .txt file for sharing")
    parser.add_argument("--mssql-conn", required=True, help="ODBC connection string")
    parser.add_argument("--out", default="schema_export.txt", help="Output .txt path")
    parser.add_argument("--include-views", action="store_true", help="Include views in the export")
    args = parser.parse_args()

    export_schema_txt(args.mssql_conn, args.out, include_views=args.include_views)


if __name__ == "__main__":
    main()

