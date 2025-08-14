"""
Unified RAG API - Consolidated Document and Database RAG System
Integrates RAGFlow for document search and database RAG for SQL queries
"""

import asyncio
import logging
import json
import threading
import os
from typing import List, Dict, Any, Optional, Union, AsyncGenerator
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
from pathlib import Path

import requests
import aiohttp
import pyodbc
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class RAGFlowConfig:
    """RAGFlow configuration"""
    url: str
    api_key: str
    default_dataset_ids: Optional[List[str]] = None
    default_chat_id: Optional[str] = None

@dataclass
class DatabaseConfig:
    """Database configuration"""
    connection_string: str
    company_prefix: str
    max_results: int = 100
    timeout: int = 30

@dataclass
class LLMConfig:
    """LLM configuration"""
    ollama_base: str
    model_name: str = "qwen2.5:7b"
    temperature: float = 0.1
    max_tokens: int = 2048

@dataclass
class APIConfig:
    """API configuration"""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    cors_origins: List[str] = None

@dataclass
class DocumentRAGConfig:
    """Main configuration for Document RAG System"""
    ragflow: RAGFlowConfig
    database: DatabaseConfig
    llm: LLMConfig
    api: APIConfig

def load_config() -> DocumentRAGConfig:
    """Load configuration from environment variables"""
    
    # RAGFlow configuration
    ragflow_config = RAGFlowConfig(
        url=os.getenv("RAGFLOW_URL", "http://192.168.0.120:82"),
        api_key=os.getenv("RAGFLOW_API_KEY", "your_api_key_here"),
        default_dataset_ids=os.getenv("RAGFLOW_DEFAULT_DATASET_IDS", "").split(",") if os.getenv("RAGFLOW_DEFAULT_DATASET_IDS") else None,
        default_chat_id=os.getenv("RAGFLOW_DEFAULT_CHAT_ID")
    )
    
    # Database configuration
    database_config = DatabaseConfig(
        connection_string=os.getenv("DATABASE_CONNECTION_STRING", 
            "DRIVER={ODBC Driver 17 for SQL Server};"
            "SERVER=192.168.1.44,1433;"
            "DATABASE=SISL Live;"
            "UID=test;"
            "PWD=Test@345;"
            "TrustServerCertificate=yes;"
        ),
        company_prefix=os.getenv("COMPANY_PREFIX", "SISL$"),
        max_results=int(os.getenv("DATABASE_MAX_RESULTS", "100")),
        timeout=int(os.getenv("DATABASE_TIMEOUT", "30"))
    )
    
    # LLM configuration
    llm_config = LLMConfig(
        ollama_base=os.getenv("OLLAMA_BASE", "http://192.168.0.120:11434"),
        model_name=os.getenv("LLM_MODEL", "qwen2.5:7b"),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
        max_tokens=int(os.getenv("LLM_MAX_TOKENS", "2048"))
    )
    
    # API configuration
    api_config = APIConfig(
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        debug=os.getenv("API_DEBUG", "false").lower() == "true",
        cors_origins=os.getenv("CORS_ORIGINS", "*").split(",") if os.getenv("CORS_ORIGINS") else ["*"]
    )
    
    return DocumentRAGConfig(
        ragflow=ragflow_config,
        database=database_config,
        llm=llm_config,
        api=api_config
    )

# ============================================================================
# RAGFLOW CLIENT
# ============================================================================

class RAGFlowError(Exception):
    """Custom exception for RAGFlow API errors"""
    pass

class ChunkMethod(Enum):
    """Supported chunking methods"""
    NAIVE = "naive"
    BOOK = "book"
    EMAIL = "email"
    LAWS = "laws"
    MANUAL = "manual"
    ONE = "one"
    PAPER = "paper"
    PICTURE = "picture"
    PRESENTATION = "presentation"
    QA = "qa"
    TABLE = "table"
    TAG = "tag"

@dataclass
class DatasetConfig:
    """Configuration for dataset creation"""
    name: str
    description: Optional[str] = None
    avatar: Optional[str] = None
    embedding_model: str = "BAAI/bge-large-zh-v1.5@BAAI"
    permission: str = "me"
    chunk_method: ChunkMethod = ChunkMethod.NAIVE
    parser_config: Optional[Dict[str, Any]] = None

@dataclass
class ChatConfig:
    """Configuration for chat assistant"""
    name: str
    dataset_ids: List[str]
    avatar: Optional[str] = None
    llm: Optional[Dict[str, Any]] = None
    prompt: Optional[Dict[str, Any]] = None

class RAGFlowClient:
    """Client for interacting with RAGFlow API"""
    
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        })
        
    def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make HTTP request to RAGFlow API"""
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            
            if response.content:
                return response.json()
            return {}
            
        except requests.exceptions.RequestException as e:
            logger.error(f"RAGFlow API request failed: {e}")
            raise RAGFlowError(f"API request failed: {e}")
    
    # Dataset Management
    def create_dataset(self, config: DatasetConfig) -> Dict[str, Any]:
        """Create a new dataset"""
        payload = {
            "name": config.name,
            "description": config.description,
            "avatar": config.avatar,
            "embedding_model": config.embedding_model,
            "permission": config.permission,
            "chunk_method": config.chunk_method.value,
            "parser_config": config.parser_config or {}
        }
        return self._make_request("POST", "/api/v1/datasets", json=payload)
    
    def list_datasets(self) -> Dict[str, Any]:
        """List all datasets"""
        return self._make_request("GET", "/api/v1/datasets")
    
    def get_dataset(self, dataset_id: str) -> Dict[str, Any]:
        """Get dataset details"""
        return self._make_request("GET", f"/api/v1/datasets/{dataset_id}")
    
    def update_dataset(self, dataset_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update dataset"""
        return self._make_request("PUT", f"/api/v1/datasets/{dataset_id}", json=updates)
    
    def delete_dataset(self, dataset_id: str) -> Dict[str, Any]:
        """Delete dataset"""
        return self._make_request("DELETE", f"/api/v1/datasets/{dataset_id}")
    
    # Document Management
    def upload_document(self, dataset_id: str, file_path: str, filename: Optional[str] = None) -> Dict[str, Any]:
        """Upload document to dataset"""
        if not filename:
            filename = Path(file_path).name
            
        with open(file_path, 'rb') as f:
            files = {'file': (filename, f, 'application/octet-stream')}
            return self._make_request("POST", f"/api/v1/datasets/{dataset_id}/upload", files=files)
    
    def list_documents(self, dataset_id: str) -> Dict[str, Any]:
        """List documents in dataset"""
        return self._make_request("GET", f"/api/v1/datasets/{dataset_id}/documents")
    
    def delete_document(self, dataset_id: str, document_id: str) -> Dict[str, Any]:
        """Delete document from dataset"""
        return self._make_request("DELETE", f"/api/v1/datasets/{dataset_id}/documents/{document_id}")
    
    def parse_document(self, dataset_id: str, document_id: str) -> Dict[str, Any]:
        """Parse document in dataset"""
        return self._make_request("POST", f"/api/v1/datasets/{dataset_id}/documents/{document_id}/parse")
    
    # Chunk Management
    def get_chunks(self, dataset_id: str, document_id: Optional[str] = None, 
                   query: Optional[str] = None, limit: int = 10) -> Dict[str, Any]:
        """Get chunks from dataset"""
        params = {"limit": limit}
        if document_id:
            params["document_id"] = document_id
        if query:
            params["query"] = query
            
        return self._make_request("GET", f"/api/v1/datasets/{dataset_id}/chunks", params=params)
    
    # Chat Assistant Management
    def create_chat_assistant(self, config: ChatConfig) -> Dict[str, Any]:
        """Create chat assistant"""
        payload = {
            "name": config.name,
            "dataset_ids": config.dataset_ids,
            "avatar": config.avatar,
            "llm": config.llm or {},
            "prompt": config.prompt or {}
        }
        return self._make_request("POST", "/api/v1/chat-assistants", json=payload)
    
    def list_chat_assistants(self) -> Dict[str, Any]:
        """List chat assistants"""
        return self._make_request("GET", "/api/v1/chat-assistants")
    
    def delete_chat_assistant(self, assistant_id: str) -> Dict[str, Any]:
        """Delete chat assistant"""
        return self._make_request("DELETE", f"/api/v1/chat-assistants/{assistant_id}")
    
    # Session Management
    def create_session(self, assistant_id: str, name: Optional[str] = None) -> Dict[str, Any]:
        """Create chat session"""
        payload = {"name": name} if name else {}
        return self._make_request("POST", f"/api/v1/chat-assistants/{assistant_id}/sessions", json=payload)
    
    def list_sessions(self, assistant_id: str) -> Dict[str, Any]:
        """List sessions for assistant"""
        return self._make_request("GET", f"/api/v1/chat-assistants/{assistant_id}/sessions")
    
    def delete_session(self, assistant_id: str, session_id: str) -> Dict[str, Any]:
        """Delete session"""
        return self._make_request("DELETE", f"/api/v1/chat-assistants/{assistant_id}/sessions/{session_id}")
    
    # Chat Completion
    def chat_completion(self, assistant_id: str, session_id: str, message: str, 
                       stream: bool = False) -> Union[Dict[str, Any], AsyncGenerator[str, None]]:
        """Send chat message"""
        payload = {
            "message": message,
            "stream": stream
        }
        
        if stream:
            return self._stream_chat_completion(assistant_id, session_id, payload)
        else:
            return self._make_request("POST", f"/api/v1/chat-assistants/{assistant_id}/sessions/{session_id}/chat", json=payload)
    
    async def _stream_chat_completion(self, assistant_id: str, session_id: str, payload: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """Stream chat completion"""
        url = f"{self.base_url}/api/v1/chat-assistants/{assistant_id}/sessions/{session_id}/chat"
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=self.session.headers) as response:
                async for line in response.content:
                    if line:
                        yield line.decode('utf-8')
    
    def health_check(self) -> Dict[str, Any]:
        """Check RAGFlow server health"""
        try:
            return self._make_request("GET", "/api/v1/health")
        except Exception as e:
            return {"status": "error", "message": str(e)}

# ============================================================================
# DATABASE RAG CLIENT
# ============================================================================

class DatabaseRAGError(Exception):
    """Custom exception for database RAG errors"""
    pass

class QueryType(Enum):
    """Types of database queries"""
    SELECT = "select"
    ANALYTICS = "analytics"
    REPORT = "report"

@dataclass
class QueryResult:
    """Result from database query"""
    sql: str
    results: List[Dict[str, Any]]
    columns: List[str]
    row_count: int
    success: bool
    error: Optional[str] = None
    execution_time: Optional[float] = None

class DatabaseRAGClient:
    """Client for database querying and schema-aware RAG"""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.embeddings = None
        self.llm = None
        self.schema_info = None
        
    async def initialize(self) -> bool:
        """Initialize database client"""
        try:
            # Initialize Ollama components
            self.embeddings = OllamaEmbeddings(
                model="nomic-embed-text",
                base_url=self.config.ollama_base
            )
            
            self.llm = ChatOllama(
                model="qwen2.5:7b",
                base_url=self.config.ollama_base,
                temperature=0.1
            )
            
            # Load schema information
            self.schema_info = self._discover_schema()
            
            logger.info("✅ Database RAG client initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize database RAG client: {e}")
            return False
    
    def _discover_schema(self) -> Dict[str, Any]:
        """Discover database schema"""
        try:
            conn = pyodbc.connect(self.config.connection_string)
            cursor = conn.cursor()
            
            schema_info = {
                "tables": {},
                "relationships": [],
                "column_details": {}
            }
            
            # Get tables
            if self.config.company_prefix:
                cursor.execute("""
                    SELECT TABLE_SCHEMA, TABLE_NAME, TABLE_TYPE
                    FROM INFORMATION_SCHEMA.TABLES 
                    WHERE TABLE_TYPE = 'BASE TABLE'
                    AND TABLE_NAME NOT LIKE '$ndo$%'
                    AND TABLE_NAME LIKE ?
                    ORDER BY TABLE_NAME
                """, f"{self.config.company_prefix}%")
            else:
                cursor.execute("""
                    SELECT TABLE_SCHEMA, TABLE_NAME, TABLE_TYPE
                    FROM INFORMATION_SCHEMA.TABLES 
                    WHERE TABLE_TYPE = 'BASE TABLE'
                    AND TABLE_NAME NOT LIKE '$ndo$%'
                    ORDER BY TABLE_NAME
                """)
            
            tables = cursor.fetchall()
            
            # Process each table
            for table in tables:
                schema, table_name, table_type = table
                
                # Get columns
                cursor.execute("""
                    SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE, COLUMN_DEFAULT
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
            
            # Get relationships
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
            
            logger.info(f"✅ Schema discovery complete: {len(schema_info['tables'])} tables")
            return schema_info
            
        except Exception as e:
            logger.error(f"❌ Error discovering schema: {e}")
            return None
    
    def _classify_intent(self, question: str) -> QueryType:
        """Classify query intent"""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ["select", "find", "get", "show", "list", "what"]):
            return QueryType.SELECT
        elif any(word in question_lower for word in ["analyze", "trend", "pattern", "insight"]):
            return QueryType.ANALYTICS
        else:
            return QueryType.REPORT
    
    def _generate_sql(self, question: str, relevant_tables: List[str]) -> str:
        """Generate SQL query using LLM"""
        try:
            # Build schema context
            schema_context = "AVAILABLE TABLES AND COLUMNS:\n"
            for table_name in relevant_tables[:5]:  # Limit to top 5 tables
                if table_name in self.schema_info["tables"]:
                    table_info = self.schema_info["tables"][table_name]
                    schema_context += f"\nTABLE: [{table_info['schema']}].[{table_name}]\n"
                    schema_context += f"COLUMNS: {', '.join(table_info['columns'])}\n"
            
            # Create prompt
            prompt = ChatPromptTemplate.from_template("""
            You are a SQL expert. Generate a SQL query based on the user's question.
            
            Schema Information:
            {schema_context}
            
            User Question: {question}
            
            Generate a SQL query that answers the question. Use only the tables and columns provided.
            Return only the SQL query, no explanations.
            """)
            
            chain = prompt | self.llm | StrOutputParser()
            
            # Run synchronously for now
            loop = asyncio.get_event_loop()
            sql = loop.run_until_complete(chain.ainvoke({
                "schema_context": schema_context,
                "question": question
            }))
            
            return sql.strip()
            
        except Exception as e:
            logger.error(f"❌ Error generating SQL: {e}")
            return ""
    
    def _execute_sql(self, sql: str) -> QueryResult:
        """Execute SQL query"""
        try:
            conn = pyodbc.connect(self.config.connection_string)
            cursor = conn.cursor()
            
            start_time = datetime.now()
            cursor.execute(sql)
            
            # Get results
            if cursor.description:
                columns = [column[0] for column in cursor.description]
                rows = cursor.fetchall()
                results = [dict(zip(columns, row)) for row in rows]
                row_count = len(results)
            else:
                results = []
                columns = []
                row_count = cursor.rowcount
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            cursor.close()
            conn.close()
            
            return QueryResult(
                sql=sql,
                results=results,
                columns=columns,
                row_count=row_count,
                success=True,
                execution_time=execution_time
            )
            
        except Exception as e:
            logger.error(f"❌ Error executing SQL: {e}")
            return QueryResult(
                sql=sql,
                results=[],
                columns=[],
                row_count=0,
                success=False,
                error=str(e)
            )
    
    async def query(self, question: str, max_results: int = 10) -> QueryResult:
        """Process database query"""
        try:
            # Classify intent
            intent = self._classify_intent(question)
            
            # Find relevant tables (simplified for now)
            relevant_tables = []
            question_lower = question.lower()
            
            for table_name in self.schema_info["tables"].keys():
                table_lower = table_name.lower()
                if any(word in table_lower for word in ["customer", "sales", "invoice", "order", "payment"]):
                    if any(word in question_lower for word in ["customer", "sales", "invoice", "order", "payment"]):
                        relevant_tables.append(table_name)
                        if len(relevant_tables) >= 3:
                            break
            
            if not relevant_tables:
                # Fallback to first few tables
                relevant_tables = list(self.schema_info["tables"].keys())[:3]
            
            # Generate SQL
            sql = self._generate_sql(question, relevant_tables)
            if not sql:
                return QueryResult(
                    sql="",
                    results=[],
                    columns=[],
                    row_count=0,
                    success=False,
                    error="Failed to generate SQL query"
                )
            
            # Execute SQL
            result = self._execute_sql(sql)
            
            # Limit results
            if result.success and len(result.results) > max_results:
                result.results = result.results[:max_results]
                result.row_count = len(result.results)
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error in database query: {e}")
            return QueryResult(
                sql="",
                results=[],
                columns=[],
                row_count=0,
                success=False,
                error=str(e)
            )
    
    def health_check(self) -> Dict[str, Any]:
        """Check database connectivity"""
        try:
            conn = pyodbc.connect(self.config.connection_string)
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            cursor.close()
            conn.close()
            return {"status": "healthy", "message": "Database connection successful"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

# ============================================================================
# ORCHESTRATOR
# ============================================================================

class IntentType(Enum):
    """Query intent classification"""
    DOCUMENT_QUERY = "document_query"
    DATABASE_QUERY = "database_query"
    HYBRID_QUERY = "hybrid_query"
    GENERAL_CONVERSATION = "general_conversation"

@dataclass
class QueryContext:
    """Context for query processing"""
    question: str
    intent: IntentType
    chat_history: List[Dict[str, str]] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    include_documents: bool = True
    include_database: bool = True
    max_results: int = 10

@dataclass
class OrchestratorResult:
    """Standardized result from orchestrator"""
    answer: str
    sources: Dict[str, Any]
    intent: IntentType
    confidence: float
    metadata: Dict[str, Any]
    success: bool
    error: Optional[str] = None

class DocumentRAGOrchestrator:
    """Main orchestrator for document and database RAG"""
    
    def __init__(self, config: DocumentRAGConfig):
        self.config = config
        self.ragflow_client = None
        self.database_client = None
        self.llm = None
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self.intent_classifier = None
        self.response_fusion = None
        
    async def initialize(self) -> bool:
        """Initialize all components"""
        try:
            # Initialize RAGFlow client
            self.ragflow_client = RAGFlowClient(
                base_url=self.config.ragflow.url,
                api_key=self.config.ragflow.api_key
            )
            
            # Initialize database client
            self.database_client = DatabaseRAGClient(self.config.database)
            await self.database_client.initialize()
            
            # Initialize LLM
            self.llm = ChatOllama(
                model=self.config.llm.model_name,
                base_url=self.config.llm.ollama_base,
                temperature=self.config.llm.temperature
            )
            
            # Initialize intent classifier
            self.intent_classifier = ChatPromptTemplate.from_template("""
            Classify the user's intent based on their question.
            
            Question: {question}
            
            Classify as one of:
            - document_query: Questions about documents, files, content, knowledge base
            - database_query: Questions about data, records, tables, business information
            - hybrid_query: Questions that need both documents and database information
            - general_conversation: General chat, greetings, non-specific questions
            
            Return only the classification.
            """)
            
            # Initialize response fusion
            self.response_fusion = ChatPromptTemplate.from_template("""
            Combine information from multiple sources into a coherent answer.
            
            User Question: {question}
            
            Document Information: {document_info}
            Database Information: {database_info}
            
            Provide a comprehensive answer that addresses the user's question using all available information.
            """)
            
            logger.info("✅ Document RAG Orchestrator initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize orchestrator: {e}")
            return False
    
    async def classify_intent(self, question: str) -> IntentType:
        """Classify user intent"""
        try:
            chain = self.intent_classifier | self.llm | StrOutputParser()
            result = await chain.ainvoke({"question": question})
            
            result_lower = result.lower().strip()
            
            if "document_query" in result_lower:
                return IntentType.DOCUMENT_QUERY
            elif "database_query" in result_lower:
                return IntentType.DATABASE_QUERY
            elif "hybrid_query" in result_lower:
                return IntentType.HYBRID_QUERY
            else:
                return IntentType.GENERAL_CONVERSATION
                
        except Exception as e:
            logger.error(f"❌ Error classifying intent: {e}")
            return IntentType.GENERAL_CONVERSATION
    
    async def query_documents(self, question: str, dataset_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Query documents via RAGFlow"""
        try:
            if not dataset_ids:
                dataset_ids = self.config.ragflow.default_dataset_ids or []
            
            if not dataset_ids:
                return {"answer": "No document datasets configured", "sources": []}
            
            # For now, use the first dataset
            dataset_id = dataset_ids[0]
            
            # Get chunks
            chunks_response = self.ragflow_client.get_chunks(
                dataset_id=dataset_id,
                query=question,
                limit=5
            )
            
            if "data" in chunks_response and chunks_response["data"]:
                chunks = chunks_response["data"]
                sources = []
                
                for chunk in chunks:
                    sources.append({
                        "content": chunk.get("content", ""),
                        "document": chunk.get("document_name", ""),
                        "score": chunk.get("score", 0)
                    })
                
                # Simple answer generation
                answer = f"Found {len(sources)} relevant document sections. "
                answer += " ".join([source["content"][:200] + "..." for source in sources[:2]])
                
                return {
                    "answer": answer,
                    "sources": sources
                }
            else:
                return {"answer": "No relevant documents found", "sources": []}
                
        except Exception as e:
            logger.error(f"❌ Error querying documents: {e}")
            return {"answer": f"Error querying documents: {str(e)}", "sources": []}
    
    async def query_database(self, question: str) -> Dict[str, Any]:
        """Query database"""
        try:
            result = await self.database_client.query(question, max_results=5)
            
            if result.success:
                return {
                    "answer": f"Found {result.row_count} records. " + 
                             (f"Sample data: {str(result.results[:2])}" if result.results else "No data found."),
                    "sql": result.sql,
                    "results": result.results,
                    "columns": result.columns,
                    "row_count": result.row_count
                }
            else:
                return {
                    "answer": f"Database query failed: {result.error}",
                    "sql": result.sql,
                    "results": [],
                    "columns": [],
                    "row_count": 0
                }
                
        except Exception as e:
            logger.error(f"❌ Error querying database: {e}")
            return {"answer": f"Error querying database: {str(e)}", "results": []}
    
    async def fuse_responses(self, question: str, document_info: Dict[str, Any], 
                           database_info: Dict[str, Any]) -> str:
        """Fuse responses from multiple sources"""
        try:
            chain = self.response_fusion | self.llm | StrOutputParser()
            
            result = await chain.ainvoke({
                "question": question,
                "document_info": str(document_info),
                "database_info": str(database_info)
            })
            
            return result.strip()
            
        except Exception as e:
            logger.error(f"❌ Error fusing responses: {e}")
            # Fallback to simple concatenation
            return f"Document Information: {document_info.get('answer', 'N/A')}\n\nDatabase Information: {database_info.get('answer', 'N/A')}"
    
    async def process_query(self, context: QueryContext) -> OrchestratorResult:
        """Process user query through the orchestrator"""
        try:
            # Classify intent if not provided
            if not context.intent:
                context.intent = await self.classify_intent(context.question)
            
            document_info = {}
            database_info = {}
            
            # Query documents if needed
            if context.include_documents and context.intent in [IntentType.DOCUMENT_QUERY, IntentType.HYBRID_QUERY]:
                document_info = await self.query_documents(context.question)
            
            # Query database if needed
            if context.include_database and context.intent in [IntentType.DATABASE_QUERY, IntentType.HYBRID_QUERY]:
                database_info = await self.query_database(context.question)
            
            # Generate final answer
            if context.intent == IntentType.HYBRID_QUERY:
                answer = await self.fuse_responses(context.question, document_info, database_info)
            elif context.intent == IntentType.DOCUMENT_QUERY:
                answer = document_info.get("answer", "No document information available")
            elif context.intent == IntentType.DATABASE_QUERY:
                answer = database_info.get("answer", "No database information available")
            else:
                answer = "I'm here to help with document and database queries. How can I assist you?"
            
            return OrchestratorResult(
                answer=answer,
                sources={
                    "documents": document_info,
                    "database": database_info
                },
                intent=context.intent,
                confidence=0.8,  # Placeholder
                metadata={
                    "processing_time": 0.0,  # Placeholder
                    "sources_used": list(filter(bool, [context.include_documents, context.include_database]))
                },
                success=True
            )
            
        except Exception as e:
            logger.error(f"❌ Error processing query: {e}")
            return OrchestratorResult(
                answer="",
                sources={},
                intent=context.intent or IntentType.GENERAL_CONVERSATION,
                confidence=0.0,
                metadata={},
                success=False,
                error=str(e)
            )
    
    async def chat_completion(self, message: str, chat_id: Optional[str] = None, 
                            stream: bool = False) -> Union[str, AsyncGenerator[str, None]]:
        """Handle chat completion with streaming support"""
        try:
            context = QueryContext(
                question=message,
                intent=None,
                chat_history=self.memory.chat_memory.messages if chat_id else []
            )
            
            result = await self.process_query(context)
            
            if stream:
                async def stream_response():
                    yield f"data: {json.dumps({'type': 'start', 'content': ''})}\n\n"
                    yield f"data: {json.dumps({'type': 'content', 'content': result.answer})}\n\n"
                    yield f"data: {json.dumps({'type': 'end', 'content': ''})}\n\n"
                
                return stream_response()
            else:
                return result.answer
                
        except Exception as e:
            logger.error(f"❌ Error in chat completion: {e}")
            if stream:
                async def stream_error():
                    yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
                return stream_error()
            else:
                return f"Error: {str(e)}"
    
    def health_check(self) -> Dict[str, Any]:
        """Check health of all components"""
        try:
            ragflow_health = self.ragflow_client.health_check() if self.ragflow_client else {"status": "not_initialized"}
            database_health = self.database_client.health_check() if self.database_client else {"status": "not_initialized"}
            
            return {
                "status": "healthy" if all(h.get("status") == "healthy" for h in [ragflow_health, database_health]) else "degraded",
                "components": {
                    "ragflow": ragflow_health,
                    "database": database_health
                }
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

# Initialize FastAPI app
app = FastAPI(
    title="Unified RAG API",
    description="Document and Database RAG System with RAGFlow Integration",
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
class ChatRequest(BaseModel):
    message: str
    chat_id: Optional[str] = None
    stream: bool = False

class ChatResponse(BaseModel):
    answer: str
    intent: str
    sources: Dict[str, Any]
    success: bool
    error: Optional[str] = None

class QueryRequest(BaseModel):
    question: str
    include_documents: bool = True
    include_database: bool = True
    max_results: int = 10

class QueryResponse(BaseModel):
    answer: str
    intent: str
    sources: Dict[str, Any]
    success: bool
    error: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    components: Dict[str, Any]
    message: Optional[str] = None

class ConfigResponse(BaseModel):
    ragflow_url: str
    database_connected: bool
    llm_model: str

# Global orchestrator instance
orchestrator = None

@app.on_event("startup")
async def startup_event():
    """Initialize the orchestrator on startup"""
    global orchestrator
    
    try:
        config = load_config()
        orchestrator = DocumentRAGOrchestrator(config)
        
        success = await orchestrator.initialize()
        if not success:
            logger.error("❌ Failed to initialize orchestrator")
        else:
            logger.info("✅ API startup complete")
            
    except Exception as e:
        logger.error(f"❌ Startup error: {e}")

@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat endpoint"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        answer = await orchestrator.chat_completion(
            message=request.message,
            chat_id=request.chat_id,
            stream=False
        )
        
        return ChatResponse(
            answer=answer,
            intent="general",
            sources={},
            success=True
        )
        
    except Exception as e:
        logger.error(f"❌ Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/chat/stream")
async def chat_stream(request: ChatRequest):
    """Streaming chat endpoint"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        async def generate():
            async for chunk in await orchestrator.chat_completion(
                message=request.message,
                chat_id=request.chat_id,
                stream=True
            ):
                yield chunk
        
        return StreamingResponse(generate(), media_type="text/plain")
        
    except Exception as e:
        logger.error(f"❌ Streaming chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Query endpoint"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        context = QueryContext(
            question=request.question,
            intent=None,
            include_documents=request.include_documents,
            include_database=request.include_database,
            max_results=request.max_results
        )
        
        result = await orchestrator.process_query(context)
        
        return QueryResponse(
            answer=result.answer,
            intent=result.intent.value,
            sources=result.sources,
            success=result.success,
            error=result.error
        )
        
    except Exception as e:
        logger.error(f"❌ Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    if not orchestrator:
        return HealthResponse(
            status="not_initialized",
            components={},
            message="Orchestrator not initialized"
        )
    
    try:
        health = orchestrator.health_check()
        return HealthResponse(
            status=health["status"],
            components=health.get("components", {}),
            message=health.get("message")
        )
        
    except Exception as e:
        return HealthResponse(
            status="error",
            components={},
            message=str(e)
        )

@app.get("/api/v1/config", response_model=ConfigResponse)
async def get_config():
    """Get configuration information"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        return ConfigResponse(
            ragflow_url=orchestrator.config.ragflow.url,
            database_connected=orchestrator.database_client is not None,
            llm_model=orchestrator.config.llm.model_name
        )
        
    except Exception as e:
        logger.error(f"❌ Config error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    config = load_config()
    
    uvicorn.run(
        "api_unified:app",
        host=config.api.host,
        port=config.api.port,
        reload=config.api.debug
    ) 