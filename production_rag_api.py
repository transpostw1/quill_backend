from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uvicorn
from langchain_qdrant import QdrantVectorStore
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.prompts import ChatPromptTemplate
import pyodbc
import logging
import json
from functools import lru_cache
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Config ---
QDRANT_URL = "http://192.168.0.120:6333/"
COLLECTION_NAME = "schema_vectors"

# 🔧 Replace these with your actual MSSQL credentials
MSSQL_CONN_STR = (
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=192.168.1.44,1433;"
    "DATABASE=SISL Live;"
    "UID=test;"
    "PWD=Test@345;"
    "TrustServerCertificate=yes;"
)

# Initialize FastAPI app
app = FastAPI(
    title="Production RAG API - DB-GPT Style",
    description="Dynamic RAG API with advanced schema discovery",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class QueryRequest(BaseModel):
    question: str
    max_results: Optional[int] = 5

class QueryResponse(BaseModel):
    question: str
    generated_sql: str
    results: List[Dict[str, Any]]
    columns: List[str]
    row_count: int
    success: bool
    error: Optional[str] = None
    schema_info: Optional[Dict] = None
    execution_time: Optional[float] = None

class HealthResponse(BaseModel):
    status: str
    message: str
    schema_tables: int
    schema_relationships: int

# Global variables
qdrant = None
embeddings = None
llm = None

@lru_cache(maxsize=1)
def get_cached_schema():
    """DB-GPT style: Cache schema for performance"""
    return discover_database_schema()

def discover_database_schema():
    """DB-GPT style: Comprehensive schema discovery"""
    try:
        conn = pyodbc.connect(MSSQL_CONN_STR)
        cursor = conn.cursor()
        
        logger.info("🔍 Discovering database schema (DB-GPT style)...")
        
        schema_info = {
            "tables": {},
            "relationships": [],
            "column_details": {},
            "table_embeddings": {}  # For semantic matching
        }
        
        # Get all business tables
        cursor.execute("""
            SELECT 
                TABLE_SCHEMA,
                TABLE_NAME,
                TABLE_TYPE
            FROM INFORMATION_SCHEMA.TABLES 
            WHERE TABLE_TYPE = 'BASE TABLE'
            AND TABLE_NAME NOT LIKE '$ndo$%'
            AND TABLE_NAME NOT LIKE 'Access Control'
            AND TABLE_NAME NOT LIKE 'Active Session'
            ORDER BY TABLE_NAME
        """)
        
        tables = cursor.fetchall()
        logger.info(f"📊 Found {len(tables)} business tables")
        
        # Process each table
        for table in tables:
            schema, table_name, table_type = table
            
            # Get columns for each table
            cursor.execute("""
                SELECT 
                    COLUMN_NAME,
                    DATA_TYPE,
                    IS_NULLABLE,
                    COLUMN_DEFAULT
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
            
            # Store column details globally
            for col in columns:
                schema_info["column_details"][f"{table_name}.{col[0]}"] = {
                    "table": table_name,
                    "column": col[0],
                    "type": col[1],
                    "nullable": col[2]
                }
        
        # Get foreign key relationships
        cursor.execute("""
            SELECT 
                fk.TABLE_NAME as source_table,
                fk.COLUMN_NAME as source_column,
                pk.TABLE_NAME as target_table,
                pk.COLUMN_NAME as target_column
            FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE fk
            INNER JOIN INFORMATION_SCHEMA.REFERENTIAL_CONSTRAINTS rc 
                ON fk.CONSTRAINT_NAME = rc.CONSTRAINT_NAME
            INNER JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE pk 
                ON rc.UNIQUE_CONSTRAINT_NAME = pk.CONSTRAINT_NAME
            WHERE fk.TABLE_NAME NOT LIKE '$ndo$%'
            AND pk.TABLE_NAME NOT LIKE '$ndo$%'
        """)
        
        relationships = cursor.fetchall()
        for rel in relationships:
            schema_info["relationships"].append({
                "source_table": rel[0],
                "source_column": rel[1],
                "target_table": rel[2],
                "target_column": rel[3]
            })
        
        cursor.close()
        conn.close()
        
        logger.info(f"✅ Discovered {len(schema_info['tables'])} tables with {len(schema_info['relationships'])} relationships")
        return schema_info
        
    except Exception as e:
        logger.error(f"❌ Error discovering schema: {e}")
        return None

def generate_semantic_context(question: str, schema_info: Dict) -> str:
    """DB-GPT style: Advanced semantic schema context generation"""
    
    question_lower = question.lower()
    relevant_tables = []
    
    # Advanced keyword matching with synonyms
    semantic_keywords = {
        'customer': ['customer', 'client', 'cust', 'buyer', 'account'],
        'sales': ['sale', 'order', 'invoice', 'revenue', 'transaction', 'billing'],
        'product': ['product', 'item', 'goods', 'merchandise', 'article'],
        'payment': ['payment', 'transaction', 'billing', 'receipt'],
        'company': ['company', 'corp', 'org', 'business', 'enterprise'],
        'amount': ['amount', 'total', 'sum', 'value', 'price', 'cost']
    }
    
    # Find relevant tables based on semantic matching
    for category, keywords in semantic_keywords.items():
        if any(word in question_lower for word in keywords):
            matching_tables = [t for t in schema_info['tables'].keys() 
                             if any(word in t.lower() for word in keywords)]
            relevant_tables.extend(matching_tables)
    
    # Remove duplicates and prioritize business tables
    relevant_tables = list(set(relevant_tables))
    
    # If no specific matches, use business logic to find relevant tables
    if not relevant_tables:
        business_patterns = ['customer', 'sale', 'order', 'invoice', 'product', 'payment']
        relevant_tables = [t for t in schema_info['tables'].keys() 
                          if any(pattern in t.lower() for pattern in business_patterns)]
    
    # Limit to top 5 most relevant tables
    relevant_tables = relevant_tables[:5]
    
    # Build comprehensive context
    context_parts = []
    
    # Add table information
    for table_name in relevant_tables:
        table_info = schema_info['tables'][table_name]
        columns_str = ", ".join(table_info['columns'][:15])  # More columns for better context
        context_parts.append(f"Table: {table_name}\nColumns: {columns_str}")
    
    # Add relationships for JOIN guidance
    relevant_relationships = []
    for rel in schema_info['relationships']:
        if rel['source_table'] in relevant_tables or rel['target_table'] in relevant_tables:
            relevant_relationships.append(f"{rel['source_table']}.{rel['source_column']} -> {rel['target_table']}.{rel['target_column']}")
    
    if relevant_relationships:
        context_parts.append(f"Relationships: {'; '.join(relevant_relationships[:5])}")
    
    # Add database format guidance
    context_parts.append("Database Format: [SISL Live].[dbo].[TableName]")
    
    return "\n\n".join(context_parts)

def generate_sql_with_context(context: str, question: str) -> str:
    """DB-GPT style: Generate SQL using discovered schema context"""
    try:
        prompt = ChatPromptTemplate.from_template("""
You are a SQL expert that generates valid SQL Server T-SQL queries based on the question and discovered database schema.

IMPORTANT RULES:
1. Use the exact table names and column names from the schema
2. Use proper JOIN syntax based on the relationships provided
3. Use the correct database format: [SISL Live].[dbo].[TableName]
4. Always use square brackets for table and column names with special characters
5. Follow SQL Server T-SQL syntax
6. Use appropriate aggregation functions (SUM, COUNT, etc.) when needed
7. Include proper WHERE clauses for filtering
8. Use ORDER BY for sorting when requested

DISCOVERED SCHEMA:
{context}

USER QUESTION:
{question}

Generate only the SQL query. No explanations. Use the exact table and column names from the schema above.
""")
        chain = prompt | llm
        result = chain.invoke({"context": context, "question": question})
        return result.content.strip()
    except Exception as e:
        logger.error(f"Error generating SQL: {e}")
        raise

def initialize_components():
    """Initialize components with DB-GPT style caching"""
    global qdrant, embeddings, llm
    
    try:
        # Initialize embeddings
        embeddings = OllamaEmbeddings(
            model="nomic-embed-text",
            base_url="http://192.168.0.120:11434"
        )
        
        # Parse URL to get host and port
        url_parts = QDRANT_URL.replace("http://", "").replace("https://", "").split(":")
        host = url_parts[0]
        port = int(url_parts[1].replace("/", ""))
        
        from qdrant_client import QdrantClient
        
        client = QdrantClient(host=host, port=port)
        qdrant = QdrantVectorStore(
            collection_name=COLLECTION_NAME,
            embedding=embeddings,
            client=client
        )

        # Initialize LLM
        llm = ChatOllama(
            model="mistral:latest", 
            temperature=0.1,
            base_url="http://192.168.0.120:11434"
        )
        
        # Cache schema discovery
        schema_info = get_cached_schema()
        if not schema_info:
            logger.error("Failed to discover database schema")
            return False
        
        logger.info("✅ All components initialized successfully with DB-GPT style caching")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        return False

def run_sql_query(sql_query: str) -> tuple[List[str], List[tuple], int]:
    """Execute SQL query and return columns, rows, and row count"""
    try:
        conn = pyodbc.connect(MSSQL_CONN_STR)
        cursor = conn.cursor()
        cursor.execute(sql_query)
        
        columns = [column[0] for column in cursor.description]
        rows = cursor.fetchall()
        row_count = len(rows)
        
        cursor.close()
        conn.close()
        
        return columns, rows, row_count
    except Exception as e:
        logger.error(f"Error executing SQL query: {e}")
        raise

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    logger.info("🚀 Starting Production RAG API with DB-GPT style schema discovery...")
    if not initialize_components():
        logger.error("Failed to initialize components. API may not work properly.")

@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    schema_info = get_cached_schema()
    return HealthResponse(
        status="healthy",
        message="Production RAG API with DB-GPT style schema discovery is running",
        schema_tables=len(schema_info["tables"]) if schema_info else 0,
        schema_relationships=len(schema_info["relationships"]) if schema_info else 0
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    schema_info = get_cached_schema()
    return HealthResponse(
        status="healthy",
        message="Production RAG API with DB-GPT style schema discovery is running",
        schema_tables=len(schema_info["tables"]) if schema_info else 0,
        schema_relationships=len(schema_info["relationships"]) if schema_info else 0
    )

@app.get("/schema")
async def get_schema():
    """Get discovered database schema"""
    schema_info = get_cached_schema()
    if not schema_info:
        raise HTTPException(status_code=500, detail="Schema not discovered")
    
    return {
        "tables": list(schema_info["tables"].keys()),
        "relationships": schema_info["relationships"],
        "total_tables": len(schema_info["tables"]),
        "total_relationships": len(schema_info["relationships"])
    }

@app.post("/query", response_model=QueryResponse)
async def query_database(request: QueryRequest):
    """Main endpoint for querying the database using natural language with DB-GPT style discovery"""
    import time
    start_time = time.time()
    
    try:
        logger.info(f"🔍 Received query: {request.question}")
        
        # Get cached schema
        schema_info = get_cached_schema()
        if not schema_info:
            raise Exception("Database schema not discovered")
        
        # Generate semantic context
        schema_context = generate_semantic_context(request.question, schema_info)
        logger.info("✅ Semantic schema context generated")
        
        # Generate SQL using discovered schema
        generated_sql = generate_sql_with_context(schema_context, request.question)
        logger.info(f"✅ Generated SQL: {generated_sql}")
        
        # Execute SQL query
        columns, rows, row_count = run_sql_query(generated_sql)
        logger.info(f"✅ Query executed successfully. Found {row_count} rows")
        
        # Convert rows to list of dictionaries
        results = []
        for row in rows:
            row_dict = {}
            for i, value in enumerate(row):
                if hasattr(value, 'isoformat'):
                    row_dict[columns[i]] = value.isoformat()
                else:
                    row_dict[columns[i]] = str(value) if value is not None else None
            results.append(row_dict)
        
        execution_time = time.time() - start_time
        
        return QueryResponse(
            question=request.question,
            generated_sql=generated_sql,
            results=results,
            columns=columns,
            row_count=row_count,
            success=True,
            schema_info={
                "tables_used": list(set([col.split('.')[0] for col in columns if '.' in col])),
                "schema_context": schema_context
            },
            execution_time=execution_time
        )
        
    except Exception as e:
        logger.error(f"❌ Error processing query: {e}")
        execution_time = time.time() - start_time
        return QueryResponse(
            question=request.question,
            generated_sql="",
            results=[],
            columns=[],
            row_count=0,
            success=False,
            error=str(e),
            execution_time=execution_time
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)  # Different port for production version 