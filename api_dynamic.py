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
    title="SISL RAG API - Dynamic",
    description="Dynamic RAG API that discovers database schema automatically",
    version="1.0.0"
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

class HealthResponse(BaseModel):
    status: str
    message: str

# Global variables
qdrant = None
embeddings = None
llm = None
db_schema = None

def get_database_schema():
    """Dynamically discover database schema - like DB-GPT does"""
    try:
        conn = pyodbc.connect(MSSQL_CONN_STR)
        cursor = conn.cursor()
        
        schema_info = {
            "tables": {},
            "relationships": [],
            "column_details": {}
        }
        
        # Get all tables
        cursor.execute("""
            SELECT 
                TABLE_SCHEMA,
                TABLE_NAME,
                TABLE_TYPE
            FROM INFORMATION_SCHEMA.TABLES 
            WHERE TABLE_TYPE = 'BASE TABLE'
            ORDER BY TABLE_NAME
        """)
        
        tables = cursor.fetchall()
        
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
        
        logger.info(f"Discovered {len(schema_info['tables'])} tables with {len(schema_info['relationships'])} relationships")
        return schema_info
        
    except Exception as e:
        logger.error(f"Error discovering schema: {e}")
        return None

def generate_schema_context(question: str, schema_info: Dict) -> str:
    """Generate relevant schema context based on the question - like DB-GPT's schema retrieval"""
    
    # Extract key terms from question
    question_lower = question.lower()
    relevant_tables = []
    
    # Simple keyword matching (DB-GPT uses more sophisticated NLP)
    if any(word in question_lower for word in ['customer', 'client']):
        relevant_tables.extend([t for t in schema_info['tables'].keys() if 'customer' in t.lower()])
    
    if any(word in question_lower for word in ['sale', 'order', 'invoice']):
        relevant_tables.extend([t for t in schema_info['tables'].keys() if any(word in t.lower() for word in ['sale', 'order', 'invoice'])])
    
    if any(word in question_lower for word in ['product', 'item']):
        relevant_tables.extend([t for t in schema_info['tables'].keys() if any(word in t.lower() for word in ['product', 'item'])])
    
    # If no specific matches, include all tables
    if not relevant_tables:
        relevant_tables = list(schema_info['tables'].keys())
    
    # Build schema context
    context_parts = []
    for table_name in relevant_tables[:5]:  # Limit to top 5 relevant tables
        table_info = schema_info['tables'][table_name]
        columns_str = ", ".join(table_info['columns'])
        context_parts.append(f"Table: {table_name}\nColumns: {columns_str}")
    
    # Add relationships
    relevant_relationships = []
    for rel in schema_info['relationships']:
        if rel['source_table'] in relevant_tables or rel['target_table'] in relevant_tables:
            relevant_relationships.append(f"{rel['source_table']}.{rel['source_column']} -> {rel['target_table']}.{rel['target_column']}")
    
    if relevant_relationships:
        context_parts.append(f"Relationships: {'; '.join(relevant_relationships[:3])}")
    
    return "\n\n".join(context_parts)

def generate_sql_dynamic(context: str, question: str, schema_info: Dict) -> str:
    """Generate SQL using dynamic schema discovery - like DB-GPT"""
    try:
        prompt = ChatPromptTemplate.from_template("""
You are a SQL expert that generates valid SQL Server T-SQL queries based on the question and discovered database schema.

IMPORTANT RULES:
1. Use the exact table names and column names from the schema
2. Use proper JOIN syntax based on the relationships provided
3. Use the correct database format: [SISL Live].[dbo].[TableName]
4. Always use square brackets for table and column names with special characters
5. Follow SQL Server T-SQL syntax

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
    """Initialize components with dynamic schema discovery"""
    global qdrant, embeddings, llm, db_schema
    
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
        
        # Discover database schema dynamically
        db_schema = get_database_schema()
        if not db_schema:
            logger.error("Failed to discover database schema")
            return False
        
        logger.info("All components initialized successfully with dynamic schema discovery")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        return False

def search_schema(query: str, k: int = 5) -> str:
    """Search schema context from Qdrant"""
    try:
        docs = qdrant.similarity_search(query, k=k)
        return "\n\n".join([doc.page_content for doc in docs])
    except Exception as e:
        logger.error(f"Error searching schema: {e}")
        raise

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
    logger.info("Starting SISL RAG API with dynamic schema discovery...")
    if not initialize_components():
        logger.error("Failed to initialize components. API may not work properly.")

@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        message="SISL RAG API with dynamic schema discovery is running"
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        message="SISL RAG API with dynamic schema discovery is running"
    )

@app.get("/schema")
async def get_schema():
    """Get discovered database schema"""
    if not db_schema:
        raise HTTPException(status_code=500, detail="Schema not discovered")
    
    return {
        "tables": list(db_schema["tables"].keys()),
        "relationships": db_schema["relationships"],
        "total_tables": len(db_schema["tables"]),
        "total_relationships": len(db_schema["relationships"])
    }

@app.post("/query", response_model=QueryResponse)
async def query_database(request: QueryRequest):
    """Main endpoint for querying the database using natural language with dynamic schema"""
    try:
        logger.info(f"Received query: {request.question}")
        
        if not db_schema:
            raise Exception("Database schema not discovered")
        
        # Generate dynamic schema context
        schema_context = generate_schema_context(request.question, db_schema)
        logger.info("Dynamic schema context generated")
        
        # Generate SQL using dynamic schema
        generated_sql = generate_sql_dynamic(schema_context, request.question, db_schema)
        logger.info(f"Generated SQL: {generated_sql}")
        
        # Execute SQL query
        columns, rows, row_count = run_sql_query(generated_sql)
        logger.info(f"Query executed successfully. Found {row_count} rows")
        
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
            }
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return QueryResponse(
            question=request.question,
            generated_sql="",
            results=[],
            columns=[],
            row_count=0,
            success=False,
            error=str(e)
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)  # Different port to test 