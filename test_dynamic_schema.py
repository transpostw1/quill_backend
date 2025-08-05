import pyodbc
import json

# Database connection
MSSQL_CONN_STR = (
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=192.168.1.44,1433;"
    "DATABASE=SISL Live;"
    "UID=test;"
    "PWD=Test@345;"
    "TrustServerCertificate=yes;"
)

def discover_schema():
    """Like DB-GPT: Dynamically discover database schema"""
    try:
        conn = pyodbc.connect(MSSQL_CONN_STR)
        cursor = conn.cursor()
        
        print("🔍 Discovering database schema (DB-GPT style)...")
        
        # Look for specific business tables we know exist
        target_tables = [
            'ssil_UAT$Customer',
            'ssil_UAT$Sales Invoice Header', 
            'ssil_UAT$Sales Invoice Line',
            'SISL$Customer',
            'SISL$Company'
        ]
        
        schema_info = {
            "tables": {},
            "relationships": [],
            "column_details": {}
        }
        
        # Check each target table
        for target_table in target_tables:
            cursor.execute("""
                SELECT 
                    TABLE_SCHEMA,
                    TABLE_NAME,
                    TABLE_TYPE
                FROM INFORMATION_SCHEMA.TABLES 
                WHERE TABLE_NAME = ?
                AND TABLE_TYPE = 'BASE TABLE'
            """, target_table)
            
            table_exists = cursor.fetchone()
            if table_exists:
                schema, table_name, table_type = table_exists
                
                # Get columns for this table
                cursor.execute("""
                    SELECT 
                        COLUMN_NAME,
                        DATA_TYPE,
                        IS_NULLABLE
                    FROM INFORMATION_SCHEMA.COLUMNS 
                    WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?
                    ORDER BY ORDINAL_POSITION
                """, schema, table_name)
                
                columns = cursor.fetchall()
                
                schema_info["tables"][table_name] = {
                    "schema": schema,
                    "columns": [col[0] for col in columns],
                    "column_details": {col[0]: {"type": col[1], "nullable": col[2]} for col in columns}
                }
                
                print(f"✅ Found: {table_name}")
                print(f"   Columns: {', '.join([col[0] for col in columns[:10]])}{'...' if len(columns) > 10 else ''}")
        
        # Also get some other business tables
        cursor.execute("""
            SELECT 
                TABLE_SCHEMA,
                TABLE_NAME,
                TABLE_TYPE
            FROM INFORMATION_SCHEMA.TABLES 
            WHERE TABLE_TYPE = 'BASE TABLE'
            AND (TABLE_NAME LIKE '%Customer%' 
                 OR TABLE_NAME LIKE '%Sales%' 
                 OR TABLE_NAME LIKE '%Invoice%'
                 OR TABLE_NAME LIKE '%Order%')
            AND TABLE_NAME NOT LIKE '$ndo$%'
            ORDER BY TABLE_NAME
        """)
        
        additional_tables = cursor.fetchall()
        print(f"\n📊 Found {len(additional_tables)} additional business tables")
        
        for table in additional_tables[:10]:
            schema, table_name, table_type = table
            
            if table_name not in schema_info["tables"]:
                # Get columns for this table
                cursor.execute("""
                    SELECT 
                        COLUMN_NAME,
                        DATA_TYPE,
                        IS_NULLABLE
                    FROM INFORMATION_SCHEMA.COLUMNS 
                    WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?
                    ORDER BY ORDINAL_POSITION
                """, schema, table_name)
                
                columns = cursor.fetchall()
                
                schema_info["tables"][table_name] = {
                    "schema": schema,
                    "columns": [col[0] for col in columns],
                    "column_details": {col[0]: {"type": col[1], "nullable": col[2]} for col in columns}
                }
                
                print(f"📋 Table: {table_name}")
                print(f"   Columns: {', '.join([col[0] for col in columns[:5]])}{'...' if len(columns) > 5 else ''}")
        
        cursor.close()
        conn.close()
        
        return schema_info
        
    except Exception as e:
        print(f"❌ Error discovering schema: {e}")
        return None

def generate_schema_context(question: str, schema_info: dict) -> str:
    """Like DB-GPT: Generate relevant schema context based on question"""
    
    print(f"\n🤖 Analyzing question: '{question}'")
    
    question_lower = question.lower()
    relevant_tables = []
    
    # Enhanced keyword matching (DB-GPT uses more sophisticated NLP)
    keywords = {
        'customer': ['customer', 'client', 'cust'],
        'sales': ['sale', 'order', 'invoice', 'revenue', 'ssil_uat$sales'],
        'product': ['product', 'item', 'goods'],
        'payment': ['payment', 'transaction', 'billing'],
        'company': ['company', 'corp', 'org']
    }
    
    for category, words in keywords.items():
        if any(word in question_lower for word in words):
            matching_tables = [t for t in schema_info['tables'].keys() 
                             if any(word in t.lower() for word in words)]
            relevant_tables.extend(matching_tables)
            if matching_tables:
                print(f"   Found {category} tables: {matching_tables[:3]}")
    
    # Remove duplicates and limit
    relevant_tables = list(set(relevant_tables))[:5]
    
    if not relevant_tables:
        # Look for tables with common business patterns
        business_tables = [t for t in schema_info['tables'].keys() 
                          if any(pattern in t.lower() for pattern in ['customer', 'sale', 'order', 'invoice', 'product'])]
        relevant_tables = business_tables[:5]
        print(f"   Using business tables: {relevant_tables}")
    
    # Build context
    context_parts = []
    for table_name in relevant_tables:
        table_info = schema_info['tables'][table_name]
        columns_str = ", ".join(table_info['columns'][:10])  # First 10 columns
        context_parts.append(f"Table: {table_name}\nColumns: {columns_str}")
    
    # Add relationships
    relevant_relationships = []
    for rel in schema_info['relationships']:
        if rel['source_table'] in relevant_tables or rel['target_table'] in relevant_tables:
            relevant_relationships.append(f"{rel['source_table']}.{rel['source_column']} -> {rel['target_table']}.{rel['target_column']}")
    
    if relevant_relationships:
        context_parts.append(f"Relationships: {'; '.join(relevant_relationships[:3])}")
    
    context = "\n\n".join(context_parts)
    print(f"\n📝 Generated schema context:\n{context}")
    
    return context

def test_dynamic_queries():
    """Test dynamic schema discovery with different queries"""
    
    schema_info = discover_schema()
    if not schema_info:
        print("❌ Failed to discover schema")
        return
    
    # Test different types of queries
    test_questions = [
        "Show me top customers by sales",
        "How many customers do I have?",
        "What are my recent orders?",
        "Count total sales this month"
    ]
    
    print("\n" + "="*50)
    print("🧪 TESTING DYNAMIC SCHEMA DISCOVERY")
    print("="*50)
    
    for question in test_questions:
        print(f"\n🔍 Question: {question}")
        context = generate_schema_context(question, schema_info)
        print(f"✅ Context generated successfully")

if __name__ == "__main__":
    test_dynamic_queries() 