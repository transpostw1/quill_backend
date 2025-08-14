#!/usr/bin/env python3
"""
Extract historical SQL queries from SQL Server for RAG training data.
Supports multiple extraction methods and outputs in TXT and YAML formats.
"""

import pyodbc
import yaml
import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import argparse
import os

class SQLHistoryExtractor:
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.queries = []
        self.sql_server_version = None
        
    def connect(self):
        """Establish database connection"""
        try:
            self.conn = pyodbc.connect(self.connection_string)
            print("✅ Connected to SQL Server successfully")
            
            # Get SQL Server version
            cursor = self.conn.cursor()
            cursor.execute("SELECT @@VERSION")
            version_info = cursor.fetchone()[0]
            print(f"📋 SQL Server Version: {version_info.split()[2] if len(version_info.split()) > 2 else 'Unknown'}")
            
            # Extract version number
            version_match = re.search(r'(\d+)\.(\d+)\.(\d+)', version_info)
            if version_match:
                self.sql_server_version = tuple(map(int, version_match.groups()))
                print(f"📋 Version tuple: {self.sql_server_version}")
            
            return True
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False
    
    def check_column_exists(self, table_name: str, column_name: str) -> bool:
        """Check if a column exists in a table"""
        try:
            cursor = self.conn.cursor()
            cursor.execute(f"""
                SELECT COUNT(*) 
                FROM INFORMATION_SCHEMA.COLUMNS 
                WHERE TABLE_NAME = '{table_name}' 
                AND COLUMN_NAME = '{column_name}'
            """)
            return cursor.fetchone()[0] > 0
        except:
            return False
    
    def extract_query_store_queries(self, start_date: str = "2024-04-01", end_date: str = "2025-04-30", min_executions: int = 1) -> List[Dict[str, Any]]:
        """Extract queries from Query Store (SQL Server 2016+)"""
        print(f"🔍 Extracting queries from Query Store ({start_date} to {end_date}, min {min_executions} executions)...")
        
        # Check if Query Store is enabled
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM sys.query_store_query_text")
            print("✅ Query Store is available")
        except Exception as e:
            print(f"⚠️ Query Store not available: {e}")
            return []
        
        # Build query based on available columns
        base_query = """
        SELECT 
            qsqt.query_sql_text,
            qsrs.execution_count,
            qsrs.last_execution_time
        FROM sys.query_store_query_text qsqt
        JOIN sys.query_store_query qsq ON qsqt.query_text_id = qsq.query_text_id
        JOIN sys.query_store_plan qsp ON qsq.query_id = qsp.query_id
        JOIN sys.query_store_runtime_stats qsrs ON qsp.plan_id = qsrs.plan_id
        WHERE qsrs.last_execution_time >= ?
        AND qsrs.last_execution_time <= ?
        AND qsrs.execution_count >= ?
        AND qsqt.query_sql_text LIKE '%SISL$%'
        ORDER BY qsrs.execution_count DESC, qsrs.last_execution_time DESC
        """
        
        # Add optional columns if they exist
        optional_columns = [
            ("qsrs.avg_duration", "avg_duration_ms"),
            ("qsrs.avg_cpu_time", "avg_cpu_time"),
            ("qsrs.avg_logical_io_reads", "avg_logical_reads")
        ]
        
        select_clause = "SELECT qsqt.query_sql_text, qsrs.execution_count, qsrs.last_execution_time"
        for col, alias in optional_columns:
            if self.check_column_exists("query_store_runtime_stats", col.split('.')[1]):
                select_clause += f", {col} as {alias}"
        
        query = base_query.replace("SELECT \n            qsqt.query_sql_text,\n            qsrs.execution_count,\n            qsrs.last_execution_time", select_clause)
        
        try:
            cursor = self.conn.cursor()
            cursor.execute(query, (start_date, end_date, min_executions))
            rows = cursor.fetchall()
            
            queries = []
            for row in rows:
                query_data = {
                    'sql_text': row[0],
                    'execution_count': row[1],
                    'last_execution': row[2].isoformat() if row[2] else None,
                    'source': 'query_store'
                }
                
                # Add optional columns if they exist
                if len(row) > 3:
                    query_data['avg_duration_ms'] = row[3] if len(row) > 3 else None
                if len(row) > 4:
                    query_data['avg_cpu_time'] = row[4] if len(row) > 4 else None
                if len(row) > 5:
                    query_data['avg_logical_reads'] = row[5] if len(row) > 5 else None
                
                queries.append(query_data)
            
            print(f"✅ Found {len(queries)} queries in Query Store")
            return queries
            
        except Exception as e:
            print(f"⚠️ Query Store extraction failed: {e}")
            return []
    
    def extract_dmv_queries(self, start_date: str = "2024-04-01", end_date: str = "2025-04-30") -> List[Dict[str, Any]]:
        """Extract queries from DMVs (recent activity)"""
        print(f"🔍 Extracting queries from DMVs ({start_date} to {end_date})...")
        
        # Build query based on available columns
        base_query = """
        SELECT 
            qs.sql_handle,
            qs.execution_count,
            qs.last_execution_time,
            st.text as sql_text
        FROM sys.dm_exec_query_stats qs
        CROSS APPLY sys.dm_exec_sql_text(qs.sql_handle) st
        WHERE st.text LIKE '%SISL$%'
        AND qs.last_execution_time >= ?
        AND qs.last_execution_time <= ?
        ORDER BY qs.last_execution_time DESC
        """
        
        # Add optional columns if they exist
        optional_columns = [
            ("qs.total_elapsed_time", "total_elapsed_time"),
            ("qs.total_cpu_time", "total_cpu_time"),
            ("qs.total_logical_reads", "total_logical_reads")
        ]
        
        select_clause = "SELECT qs.sql_handle, qs.execution_count, qs.last_execution_time"
        for col, alias in optional_columns:
            if self.check_column_exists("dm_exec_query_stats", col.split('.')[1]):
                select_clause += f", {col} as {alias}"
        
        query = base_query.replace("SELECT \n            qs.sql_handle,\n            qs.execution_count,\n            qs.last_execution_time", select_clause)
        query += ", st.text as sql_text"
        
        try:
            cursor = self.conn.cursor()
            cursor.execute(query, (start_date, end_date))
            rows = cursor.fetchall()
            
            queries = []
            for row in rows:
                query_data = {
                    'sql_text': row[-1],  # sql_text is always last
                    'execution_count': row[1],
                    'last_execution': row[2].isoformat() if row[2] else None,
                    'source': 'dmv'
                }
                
                # Add optional columns if they exist
                if len(row) > 4:
                    query_data['total_elapsed_time'] = row[3] if len(row) > 3 else None
                if len(row) > 5:
                    query_data['total_cpu_time'] = row[4] if len(row) > 4 else None
                if len(row) > 6:
                    query_data['total_logical_reads'] = row[5] if len(row) > 5 else None
                
                queries.append(query_data)
            
            print(f"✅ Found {len(queries)} queries in DMVs")
            return queries
            
        except Exception as e:
            print(f"⚠️ DMV extraction failed: {e}")
            return []
    
    def extract_nav_activity_queries(self) -> List[Dict[str, Any]]:
        """Extract queries by analyzing NAV table activity patterns (works with basic permissions)"""
        print("🔍 Analyzing NAV table activity patterns...")
        
        # Get list of NAV tables and their recent activity
        nav_tables_query = """
        SELECT 
            TABLE_NAME,
            TABLE_SCHEMA
        FROM INFORMATION_SCHEMA.TABLES 
        WHERE TABLE_NAME LIKE '%SISL$%'
        AND TABLE_TYPE = 'BASE TABLE'
        ORDER BY TABLE_NAME
        """
        
        try:
            cursor = self.conn.cursor()
            cursor.execute(nav_tables_query)
            nav_tables = cursor.fetchall()
            
            print(f"✅ Found {len(nav_tables)} NAV tables")
            
            queries = []
            
            # Generate common query patterns for each table
            for table_schema, table_name in nav_tables:
                full_table_name = f"[{table_schema}].[{table_name}]"
                
                # Get column info for this table
                columns_query = f"""
                SELECT COLUMN_NAME, DATA_TYPE 
                FROM INFORMATION_SCHEMA.COLUMNS 
                WHERE TABLE_NAME = '{table_name}' 
                AND TABLE_SCHEMA = '{table_schema}'
                ORDER BY ORDINAL_POSITION
                """
                
                cursor.execute(columns_query)
                columns = cursor.fetchall()
                
                # Generate common query patterns
                table_queries = self.generate_table_queries(full_table_name, columns)
                queries.extend(table_queries)
            
            print(f"✅ Generated {len(queries)} query patterns from NAV tables")
            return queries
            
        except Exception as e:
            print(f"⚠️ NAV activity analysis failed: {e}")
            return []
    
    def generate_table_queries(self, table_name: str, columns: List[tuple]) -> List[Dict[str, Any]]:
        """Generate common query patterns for a NAV table"""
        queries = []
        
        # Extract column names and types
        column_names = [col[0] for col in columns]
        column_types = {col[0]: col[1] for col in columns}
        
        # Common patterns based on table name
        table_lower = table_name.lower()
        
        # Count queries
        if any(keyword in table_lower for keyword in ['customer', 'vendor', 'item', 'employee']):
            sql = f"SELECT COUNT(*) as TotalCount FROM {table_name}"
            queries.append({
                'sql_text': sql,
                'natural_language': f"Count records from {table_name.split('$')[-1].replace(']', '')}",
                'source': 'generated_pattern',
                'execution_count': 1,
                'last_execution': datetime.now().isoformat()
            })
        
        # Date-based queries (if table has date columns)
        date_columns = [col for col in column_names if any(date_type in column_types[col].lower() 
                                                         for date_type in ['date', 'datetime', 'timestamp'])]
        
        if date_columns:
            # Use the first date column
            date_col = date_columns[0]
            sql = f"SELECT TOP 10 * FROM {table_name} WHERE {date_col} >= DATEADD(day, -30, GETDATE()) ORDER BY {date_col} DESC"
            queries.append({
                'sql_text': sql,
                'natural_language': f"Show recent {table_name.split('$')[-1].replace(']', '')} data",
                'source': 'generated_pattern',
                'execution_count': 1,
                'last_execution': datetime.now().isoformat()
            })
        
        # Key field queries (common NAV patterns)
        key_fields = ['No_', 'Code', 'ID', 'Number']
        for key_field in key_fields:
            if key_field in column_names:
                sql = f"SELECT TOP 10 {key_field}, * FROM {table_name} ORDER BY {key_field}"
                queries.append({
                    'sql_text': sql,
                    'natural_language': f"List {table_name.split('$')[-1].replace(']', '')} records",
                    'source': 'generated_pattern',
                    'execution_count': 1,
                    'last_execution': datetime.now().isoformat()
                })
                break
        
        # Join patterns for related tables
        if 'Customer' in table_name:
            # Look for related customer tables
            related_tables = ['Sales Header', 'Sales Line', 'Customer Ledger Entry']
            for related in related_tables:
                if related in table_name:
                    continue
                # Generate join query
                sql = f"""
                SELECT TOP 10 c.[No_], c.[Name], s.[Document No_], s.[Posting Date]
                FROM {table_name} c
                LEFT JOIN [dbo].[SISL$Sales Header] s ON c.[No_] = s.[Sell-to Customer No_]
                WHERE s.[Posting Date] >= DATEADD(day, -90, GETDATE())
                ORDER BY s.[Posting Date] DESC
                """
                queries.append({
                    'sql_text': sql,
                    'natural_language': f"Show {table_name.split('$')[-1].replace(']', '')} with recent sales",
                    'source': 'generated_pattern',
                    'execution_count': 1,
                    'last_execution': datetime.now().isoformat()
                })
                break
        
        return queries
    
    def extract_simple_queries(self) -> List[Dict[str, Any]]:
        """Extract queries using a simpler approach that works on older SQL Server versions"""
        print("🔍 Trying simple query extraction...")
        
        # Try to get recent queries from plan cache
        query = """
        SELECT TOP 100
            st.text as sql_text,
            qs.execution_count,
            qs.last_execution_time
        FROM sys.dm_exec_query_stats qs
        CROSS APPLY sys.dm_exec_sql_text(qs.sql_handle) st
        WHERE st.text LIKE '%SISL$%'
        AND st.text NOT LIKE '%sys.%'
        AND st.text NOT LIKE '%INFORMATION_SCHEMA%'
        ORDER BY qs.last_execution_time DESC
        """
        
        try:
            cursor = self.conn.cursor()
            cursor.execute(query)
            rows = cursor.fetchall()
            
            queries = []
            for row in rows:
                queries.append({
                    'sql_text': row[0],
                    'execution_count': row[1],
                    'last_execution': row[2].isoformat() if row[2] else None,
                    'source': 'simple_cache'
                })
            
            print(f"✅ Found {len(queries)} queries using simple extraction")
            return queries
            
        except Exception as e:
            print(f"⚠️ Simple extraction failed: {e}")
            return []
    
    def clean_and_deduplicate_queries(self, queries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Clean and deduplicate queries"""
        print("🧹 Cleaning and deduplicating queries...")
        
        # Clean SQL text
        for query in queries:
            sql = query['sql_text']
            if sql:
                # Remove extra whitespace
                sql = re.sub(r'\s+', ' ', sql.strip())
                # Remove comments
                sql = re.sub(r'--.*$', '', sql, flags=re.MULTILINE)
                sql = re.sub(r'/\*.*?\*/', '', sql, flags=re.DOTALL)
                query['sql_text'] = sql.strip()
        
        # Deduplicate by SQL text
        seen = set()
        unique_queries = []
        
        for query in queries:
            sql = query['sql_text']
            if sql and sql not in seen:
                seen.add(sql)
                unique_queries.append(query)
        
        print(f"✅ Deduplicated to {len(unique_queries)} unique queries")
        return unique_queries
    
    def generate_natural_language(self, sql: str) -> str:
        """Generate natural language description from SQL"""
        sql_upper = sql.upper()
        
        # Extract table names
        table_pattern = r'\[SISL\$([^\]]+)\]'
        tables = re.findall(table_pattern, sql)
        
        # Extract common patterns
        if 'COUNT(*)' in sql_upper:
            if 'CUSTOMER' in sql_upper:
                return "How many customers do we have?"
            elif 'VENDOR' in sql_upper:
                return "How many vendors do we have?"
            else:
                return f"Count records from {', '.join(tables)}"
        
        elif 'SELECT' in sql_upper and 'JOIN' in sql_upper:
            return f"Show data from {', '.join(tables)} with related information"
        
        elif 'WHERE' in sql_upper and any(word in sql_upper for word in ['DATE', 'TIME']):
            return f"Show {', '.join(tables)} data for specific date range"
        
        elif 'ORDER BY' in sql_upper:
            return f"Show {', '.join(tables)} data sorted by criteria"
        
        else:
            return f"Query data from {', '.join(tables)}"
    
    def validate_queries(self, queries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate queries by testing them against the database"""
        print("🔍 Validating generated queries...")
        
        validated_queries = []
        successful_count = 0
        failed_count = 0
        
        for i, query in enumerate(queries, 1):
            sql = query['sql_text']
            print(f"Testing query {i}/{len(queries)}: {sql[:50]}...", end=" ")
            
            try:
                cursor = self.conn.cursor()
                cursor.execute(sql)
                
                # Try to fetch a few rows to ensure it works
                rows = cursor.fetchmany(5)
                row_count = len(rows)
                
                # Mark as successful
                query['validation'] = {
                    'status': 'success',
                    'row_count': row_count,
                    'sample_data': rows if rows else []
                }
                successful_count += 1
                print("✅ SUCCESS")
                
            except Exception as e:
                # Mark as failed
                query['validation'] = {
                    'status': 'failed',
                    'error': str(e),
                    'row_count': 0,
                    'sample_data': []
                }
                failed_count += 1
                print(f"❌ FAILED: {str(e)[:50]}...")
            
            validated_queries.append(query)
        
        print(f"\n📊 VALIDATION SUMMARY:")
        print(f"  ✅ Successful: {successful_count}")
        print(f"  ❌ Failed: {failed_count}")
        print(f"  📈 Success rate: {(successful_count/len(queries)*100):.1f}%")
        
        return validated_queries
    
    def save_to_txt(self, filename: str, queries: List[Dict[str, Any]]):
        """Save queries to TXT file"""
        print(f"💾 Saving {len(queries)} queries to {filename}...")
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("SQL QUERY TRAINING DATA\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"Total queries: {len(queries)}\n")
            f.write(f"SQL Server Version: {self.sql_server_version}\n\n")
            
            # Count validation results
            successful = sum(1 for q in queries if q.get('validation', {}).get('status') == 'success')
            failed = sum(1 for q in queries if q.get('validation', {}).get('status') == 'failed')
            f.write(f"Validation Results: {successful} successful, {failed} failed\n\n")
            
            for i, query in enumerate(queries, 1):
                f.write(f"QUERY #{i}\n")
                f.write("-" * 30 + "\n")
                f.write(f"Source: {query['source']}\n")
                f.write(f"Executions: {query.get('execution_count', 'N/A')}\n")
                f.write(f"Last executed: {query.get('last_execution', 'N/A')}\n")
                f.write(f"Natural language: {self.generate_natural_language(query['sql_text'])}\n")
                
                # Add validation info
                validation = query.get('validation', {})
                if validation:
                    status = validation.get('status', 'unknown')
                    f.write(f"Validation: {status.upper()}\n")
                    if status == 'success':
                        f.write(f"Row count: {validation.get('row_count', 0)}\n")
                    else:
                        f.write(f"Error: {validation.get('error', 'Unknown error')}\n")
                
                f.write(f"SQL:\n{query['sql_text']}\n\n")
        
        print(f"✅ Saved to {filename}")
    
    def save_to_yaml(self, filename: str, queries: List[Dict[str, Any]]):
        """Save queries to YAML file"""
        print(f"💾 Saving {len(queries)} queries to {filename}...")
        
        # Count validation results
        successful = sum(1 for q in queries if q.get('validation', {}).get('status') == 'success')
        failed = sum(1 for q in queries if q.get('validation', {}).get('status') == 'failed')
        
        yaml_data = {
            'metadata': {
                'generated': datetime.now().isoformat(),
                'total_queries': len(queries),
                'successful_queries': successful,
                'failed_queries': failed,
                'success_rate': f"{(successful/len(queries)*100):.1f}%" if queries else "0%",
                'source': 'sql_server_history',
                'sql_server_version': str(self.sql_server_version) if self.sql_server_version else 'unknown'
            },
            'queries': []
        }
        
        for query in queries:
            yaml_query = {
                'sql': query['sql_text'],
                'natural_language': self.generate_natural_language(query['sql_text']),
                'metadata': {
                    'source': query['source'],
                    'execution_count': query.get('execution_count'),
                    'last_execution': query.get('last_execution'),
                    'avg_duration_ms': query.get('avg_duration_ms'),
                    'avg_cpu_time': query.get('avg_cpu_time')
                },
                'validation': query.get('validation', {})
            }
            yaml_data['queries'].append(yaml_query)
        
        with open(filename, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_data, f, default_flow_style=False, indent=2, allow_unicode=True)
        
        print(f"✅ Saved to {filename}")
    
    def extract_all(self, start_date: str = "2024-04-01", end_date: str = "2025-04-30", min_executions: int = 1):
        """Extract all available query data"""
        if not self.connect():
            return
        
        all_queries = []
        
        # Try Query Store first
        query_store_queries = self.extract_query_store_queries(start_date, end_date, min_executions)
        all_queries.extend(query_store_queries)
        
        # Try DMVs for recent activity
        dmv_queries = self.extract_dmv_queries(start_date, end_date)
        all_queries.extend(dmv_queries)
        
        # Try NAV activity analysis (works with basic permissions)
        nav_queries = self.extract_nav_activity_queries()
        all_queries.extend(nav_queries)
        
        # If still no queries, try simple extraction
        if not all_queries:
            simple_queries = self.extract_simple_queries()
            all_queries.extend(simple_queries)
        
        # Clean and deduplicate
        clean_queries = self.clean_and_deduplicate_queries(all_queries)
        
        if clean_queries:
            # Validate queries
            validated_queries = self.validate_queries(clean_queries)
            
            # Save in both formats
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.save_to_txt(f"sql_history_{timestamp}.txt", validated_queries)
            self.save_to_yaml(f"sql_history_{timestamp}.yaml", validated_queries)
            
            # Print summary
            print("\n📊 EXTRACTION SUMMARY:")
            print(f"Total unique queries: {len(validated_queries)}")
            
            # Count validation results
            successful = sum(1 for q in validated_queries if q.get('validation', {}).get('status') == 'success')
            failed = sum(1 for q in validated_queries if q.get('validation', {}).get('status') == 'failed')
            print(f"✅ Successful: {successful}")
            print(f"❌ Failed: {failed}")
            print(f"📈 Success rate: {(successful/len(validated_queries)*100):.1f}%")
            
            # Group by source
            sources = {}
            for q in validated_queries:
                source = q['source']
                sources[source] = sources.get(source, 0) + 1
            
            print(f"\n📋 BY SOURCE:")
            for source, count in sources.items():
                print(f"  {source}: {count} queries")
            
            # Show sample successful queries
            successful_queries = [q for q in validated_queries if q.get('validation', {}).get('status') == 'success']
            print(f"\n📝 SAMPLE SUCCESSFUL QUERIES:")
            for i, query in enumerate(successful_queries[:5], 1):
                nl = self.generate_natural_language(query['sql_text'])
                row_count = query.get('validation', {}).get('row_count', 0)
                print(f"{i}. {nl} (returns {row_count} rows)")
                print(f"   SQL: {query['sql_text'][:100]}...")
                print()
        else:
            print("❌ No queries found. This could mean:")
            print("   - Query Store is not enabled")
            print("   - No recent queries contain 'SISL$'")
            print("   - SQL Server version doesn't support these DMVs")
            print("   - No queries have been executed recently")
        
        self.conn.close()

def main():
    parser = argparse.ArgumentParser(description='Extract SQL query history from SQL Server')
    parser.add_argument('--server', required=True, help='SQL Server instance (e.g., localhost or 192.168.0.120)')
    parser.add_argument('--database', required=True, help='Database name (e.g., SISL Live)')
    parser.add_argument('--username', required=True, help='SQL Server username')
    parser.add_argument('--password', required=True, help='SQL Server password')
    parser.add_argument('--driver', default='ODBC Driver 17 for SQL Server', help='ODBC driver name')
    parser.add_argument('--start-date', default='2024-04-01', help='Start date (YYYY-MM-DD) for query extraction')
    parser.add_argument('--end-date', default='2025-04-30', help='End date (YYYY-MM-DD) for query extraction')
    parser.add_argument('--min-executions', type=int, default=1, help='Minimum execution count for Query Store')
    
    args = parser.parse_args()
    
    # Build connection string
    conn_str = (
        f"DRIVER={{{args.driver}}};"
        f"SERVER={args.server};"
        f"DATABASE={args.database};"
        f"UID={args.username};"
        f"PWD={args.password};"
        "Trusted_Connection=no;"
    )
    
    # Extract queries
    extractor = SQLHistoryExtractor(conn_str)
    extractor.extract_all(args.start_date, args.end_date, args.min_executions)

if __name__ == "__main__":
    main() 