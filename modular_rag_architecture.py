"""
Modular RAG Architecture - DB-GPT Inspired
Separates database and document RAG into distinct components with a middle orchestrator
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import asyncio
import json
import logging
from pathlib import Path
import tempfile
import os
import mimetypes

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import AsyncGenerator

from langchain_qdrant import QdrantVectorStore
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    UnstructuredFileLoader,
    PyPDFLoader,
    UnstructuredPDFLoader,
    TextLoader,
    UnstructuredExcelLoader,
    UnstructuredPowerPointLoader
)
import pyodbc
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
QDRANT_URL = "http://192.168.0.120:6333/"
COLLECTION_NAME = "schema_vectors"
DOCUMENTS_COLLECTION = "document_vectors"
OCR_API_URL = "http://192.168.0.120:3001/api/documents"

MSSQL_CONN_STR = (
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=192.168.1.44,1433;"
    "DATABASE=SISL Live;"
    "UID=test;"
    "PWD=Test@345;"
    "TrustServerCertificate=yes;"
)

class IntentType(Enum):
    DATABASE_QUERY = "database_query"
    DOCUMENT_QUERY = "document_query"
    GENERAL_CONVERSATION = "general_conversation"
    HYBRID_QUERY = "hybrid_query"

@dataclass
class QueryResult:
    """Standardized query result"""
    content: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    source: str
    success: bool
    error: Optional[str] = None

@dataclass
class QueryContext:
    """Context for query processing"""
    question: str
    intent: IntentType
    chat_history: str = ""
    include_database: bool = True
    include_documents: bool = True
    max_results: int = 5

class BaseRAGComponent(ABC):
    """Base class for RAG components"""
    
    def __init__(self, name: str):
        self.name = name
        self.embeddings = None
        self.llm = None
        self.vector_store = None
        
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the component"""
        pass
    
    @abstractmethod
    async def process_query(self, context: QueryContext) -> QueryResult:
        """Process a query and return results"""
        pass
    
    @abstractmethod
    async def is_relevant(self, question: str) -> bool:
        """Check if this component is relevant for the query"""
        pass

class DatabaseRAGComponent(BaseRAGComponent):
    """Database-specific RAG component"""
    
    def __init__(self):
        super().__init__("database_rag")
        
    async def initialize(self) -> bool:
        """Initialize database RAG components"""
        try:
            # Initialize embeddings
            self.embeddings = OllamaEmbeddings(
                model="nomic-embed-text",
                base_url="http://192.168.0.120:11434"
            )
            
            # Initialize LLM
            self.llm = ChatOllama(
                model="mistral:latest",
                temperature=0.1,
                base_url="http://192.168.0.120:11434"
            )
            
            # Initialize vector store
            url_parts = QDRANT_URL.replace("http://", "").replace("https://", "").split(":")
            host = url_parts[0]
            port = int(url_parts[1].replace("/", ""))
            
            from qdrant_client import QdrantClient
            client = QdrantClient(host=host, port=port)
            
            self.vector_store = QdrantVectorStore(
                collection_name=COLLECTION_NAME,
                embedding=self.embeddings,
                client=client
            )
            
            logger.info("Database RAG component initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize database RAG: {e}")
            return False
    
    async def is_relevant(self, question: str) -> bool:
        """Check if question is database-related"""
        try:
            if not self.llm:
                return False
                
            intent_prompt = f"""
            Determine if this question requires database access:
            "{question}"
            
            Return only: "yes" or "no"
            """
            
            result = self.llm.invoke(intent_prompt)
            return result.content.strip().lower() == "yes"
            
        except Exception as e:
            logger.error(f"Error checking database relevance: {e}")
            return False
    
    async def process_query(self, context: QueryContext) -> QueryResult:
        """Process database query"""
        try:
            # Search schema context
            docs = self.vector_store.similarity_search(context.question, k=context.max_results)
            schema_context = "\n".join([doc.page_content for doc in docs])
            
            if not schema_context:
                return QueryResult(
                    content=[],
                    metadata={"sql": None},
                    source="database",
                    success=False,
                    error="No relevant schema context found"
                )
            
            # Generate SQL
            sql = self._generate_sql(schema_context, context.question)
            if not sql:
                return QueryResult(
                    content=[],
                    metadata={"sql": None},
                    source="database",
                    success=False,
                    error="Failed to generate SQL"
                )
            
            # Execute query
            columns, rows, row_count = self._execute_sql(sql)
            
            # Format results
            results = []
            for row in rows:
                row_dict = {}
                for i, value in enumerate(row):
                    if hasattr(value, 'isoformat'):
                        row_dict[columns[i]] = value.isoformat()
                    else:
                        row_dict[columns[i]] = str(value) if value is not None else None
                results.append(row_dict)
            
            return QueryResult(
                content=results,
                metadata={
                    "sql": sql,
                    "columns": columns,
                    "row_count": row_count
                },
                source="database",
                success=True
            )
            
        except Exception as e:
            logger.error(f"Error processing database query: {e}")
            return QueryResult(
                content=[],
                metadata={},
                source="database",
                success=False,
                error=str(e)
            )
    
    def _generate_sql(self, context: str, question: str) -> str:
        """Generate SQL from context and question"""
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a SQL expert. Generate SQL queries based on the database schema context and user questions.

                Database Schema Context:
                {context}

                Rules:
                1. Use exact table and column names from the context
                2. Use proper SQL Server syntax
                3. Include proper table aliases
                4. Use appropriate JOINs when needed
                5. Return only the SQL query, no explanations"""),
                ("human", "Question: {question}")
            ])

            chain = prompt | self.llm
            result = chain.invoke({"context": context, "question": question})
            return result.content.strip()
        except Exception as e:
            logger.error(f"Error generating SQL: {e}")
            return ""
    
    def _execute_sql(self, sql_query: str) -> tuple[List[str], List[tuple], int]:
        """Execute SQL query on database"""
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
            logger.error(f"Database query error: {e}")
            raise e

class DocumentRAGComponent(BaseRAGComponent):
    """Document-specific RAG component"""
    
    def __init__(self):
        super().__init__("document_rag")
        self.text_splitter = None
        
    async def initialize(self) -> bool:
        """Initialize document RAG components"""
        try:
            # Initialize embeddings
            self.embeddings = OllamaEmbeddings(
                model="nomic-embed-text",
                base_url="http://192.168.0.120:11434"
            )
            
            # Initialize LLM
            self.llm = ChatOllama(
                model="mistral:latest",
                temperature=0.1,
                base_url="http://192.168.0.120:11434"
            )
            
            # Initialize vector store
            url_parts = QDRANT_URL.replace("http://", "").replace("https://", "").split(":")
            host = url_parts[0]
            port = int(url_parts[1].replace("/", ""))
            
            from qdrant_client import QdrantClient
            client = QdrantClient(host=host, port=port)
            
            self.vector_store = QdrantVectorStore(
                collection_name=DOCUMENTS_COLLECTION,
                embedding=self.embeddings,
                client=client
            )
            
            # Initialize text splitter
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
            )
            
            logger.info("Document RAG component initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize document RAG: {e}")
            return False
    
    async def is_relevant(self, question: str) -> bool:
        """Check if question is document-related"""
        try:
            if not self.llm:
                return False
                
            intent_prompt = f"""
            Determine if this question requires document/knowledge base access:
            "{question}"
            
            Return only: "yes" or "no"
            """
            
            result = self.llm.invoke(intent_prompt)
            return result.content.strip().lower() == "yes"
            
        except Exception as e:
            logger.error(f"Error checking document relevance: {e}")
            return False
    
    async def process_query(self, context: QueryContext) -> QueryResult:
        """Process document query"""
        try:
            # Search documents
            docs = self.vector_store.similarity_search(context.question, k=context.max_results)
            
            if not docs:
                return QueryResult(
                    content=[],
                    metadata={},
                    source="documents",
                    success=False,
                    error="No relevant documents found"
                )
            
            # Format results
            results = []
            for doc in docs:
                results.append({
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "unknown"),
                    "chunk_id": doc.metadata.get("chunk_id", 0),
                    "metadata": doc.metadata
                })
            
            return QueryResult(
                content=results,
                metadata={
                    "document_count": len(results),
                    "query": context.question
                },
                source="documents",
                success=True
            )
            
        except Exception as e:
            logger.error(f"Error processing document query: {e}")
            return QueryResult(
                content=[],
                metadata={},
                source="documents",
                success=False,
                error=str(e)
            )
    
    async def upload_document(self, file_path: str, file_type: str, description: str = None) -> Dict[str, Any]:
        """Upload and process a document"""
        try:
            documents = []
            
            if file_type and file_type.startswith('image/'):
                # Use OCR API for images
                documents = await self._process_with_ocr(file_path)
            else:
                # Use LangChain loaders
                loader = self._get_document_loader(file_path, file_type)
                if loader:
                    raw_docs = loader.load()
                    for doc in raw_docs:
                        chunks = self.text_splitter.split_documents([doc])
                        documents.extend(chunks)
            
            if documents:
                # Add to vector store
                self.vector_store.add_documents(documents)
                
                return {
                    "status": "success",
                    "message": f"Document processed successfully. Added {len(documents)} chunks.",
                    "chunks": len(documents)
                }
            else:
                return {
                    "status": "error",
                    "message": "Failed to process document. No content extracted."
                }
                
        except Exception as e:
            logger.error(f"Error uploading document: {e}")
            return {
                "status": "error",
                "message": f"Error processing document: {str(e)}"
            }
    
    def _get_document_loader(self, file_path: str, file_type: str):
        """Get appropriate document loader based on file type"""
        try:
            if file_type.startswith('image/'):
                return None  # Will handle separately
            elif file_type == 'application/pdf':
                return UnstructuredPDFLoader(file_path)
            elif file_type == 'text/plain':
                return TextLoader(file_path)
            elif file_type in ['application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                              'application/vnd.ms-excel']:
                return UnstructuredExcelLoader(file_path)
            elif file_type in ['application/vnd.openxmlformats-officedocument.presentationml.presentation',
                              'application/vnd.ms-powerpoint']:
                return UnstructuredPowerPointLoader(file_path)
            else:
                return UnstructuredFileLoader(file_path)
        except Exception as e:
            logger.error(f"Error creating loader for {file_path}: {e}")
            return None
    
    async def _process_with_ocr(self, file_path: str) -> List[Document]:
        """Process document using OCR API"""
        try:
            with open(file_path, 'rb') as f:
                files = {'file': f}
                response = requests.post(OCR_API_URL, files=files)

            if response.status_code == 200:
                result = response.json()
                text = result.get('text', '')

                # Create document chunks
                chunks = self.text_splitter.split_text(text)
                documents = []

                for i, chunk in enumerate(chunks):
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            'source': file_path,
                            'chunk_id': i,
                            'total_chunks': len(chunks),
                            'processed_by': 'ocr_api'
                        }
                    )
                    documents.append(doc)

                return documents
            else:
                logger.error(f"OCR API error: {response.status_code} - {response.text}")
                return []

        except Exception as e:
            logger.error(f"Error processing document with OCR: {e}")
            return []

class ConversationComponent(BaseRAGComponent):
    """General conversation component"""
    
    def __init__(self):
        super().__init__("conversation")
        
    async def initialize(self) -> bool:
        """Initialize conversation component"""
        try:
            self.llm = ChatOllama(
                model="mistral:latest",
                temperature=0.7,
                base_url="http://192.168.0.120:11434"
            )
            logger.info("Conversation component initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize conversation component: {e}")
            return False
    
    async def is_relevant(self, question: str) -> bool:
        """Check if question is general conversation"""
        try:
            if not self.llm:
                return False
                
            intent_prompt = f"""
            Determine if this is general conversation (greetings, casual chat):
            "{question}"
            
            Return only: "yes" or "no"
            """
            
            result = self.llm.invoke(intent_prompt)
            return result.content.strip().lower() == "yes"
            
        except Exception as e:
            logger.error(f"Error checking conversation relevance: {e}")
            return False
    
    async def process_query(self, context: QueryContext) -> QueryResult:
        """Process general conversation"""
        try:
            if not self.llm:
                return QueryResult(
                    content=[{"response": "Hello! I'm here to help you with database and document queries. How can I assist you today?"}],
                    metadata={"is_conversation": True},
                    source="conversation",
                    success=True
                )
                
            conversation_prompt = f"""
            You are a helpful assistant for a database and document querying system.
            Respond appropriately to this user input: "{context.question}"
            Keep your response concise and friendly.
            {context.chat_history}
            """

            result = self.llm.invoke(conversation_prompt)
            response_text = result.content.strip()

            return QueryResult(
                content=[{"response": response_text}],
                metadata={"is_conversation": True},
                source="conversation",
                success=True
            )
            
        except Exception as e:
            logger.error(f"Error processing conversation: {e}")
            return QueryResult(
                content=[{"response": "Hello! I'm here to help you with database and document queries. How can I assist you today?"}],
                metadata={"is_conversation": True},
                source="conversation",
                success=True
            )

class RAGOrchestrator:
    """Main orchestrator for RAG components"""
    
    def __init__(self):
        self.components = {}
        self.memory_manager = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
    async def initialize(self) -> bool:
        """Initialize all RAG components"""
        try:
            # Initialize components
            self.components["database"] = DatabaseRAGComponent()
            self.components["documents"] = DocumentRAGComponent()
            self.components["conversation"] = ConversationComponent()
            
            # Initialize each component
            for name, component in self.components.items():
                success = await component.initialize()
                if not success:
                    logger.error(f"Failed to initialize {name} component")
                    return False
            
            logger.info("All RAG components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize orchestrator: {e}")
            return False
    
    async def classify_intent(self, question: str) -> IntentType:
        """Classify user intent"""
        try:
            # Check each component's relevance
            relevant_components = []
            
            for name, component in self.components.items():
                if await component.is_relevant(question):
                    relevant_components.append(name)
            
            # Determine intent based on relevant components
            if "conversation" in relevant_components and len(relevant_components) == 1:
                return IntentType.GENERAL_CONVERSATION
            elif "database" in relevant_components and "documents" in relevant_components:
                return IntentType.HYBRID_QUERY
            elif "database" in relevant_components:
                return IntentType.DATABASE_QUERY
            elif "documents" in relevant_components:
                return IntentType.DOCUMENT_QUERY
            else:
                return IntentType.GENERAL_CONVERSATION
                
        except Exception as e:
            logger.error(f"Error classifying intent: {e}")
            return IntentType.GENERAL_CONVERSATION
    
    async def process_query(self, context: QueryContext) -> List[QueryResult]:
        """Process query using appropriate components"""
        try:
            results = []
            
            # Classify intent
            intent = await self.classify_intent(context.question)
            context.intent = intent
            
            # Process with relevant components
            if intent == IntentType.GENERAL_CONVERSATION:
                result = await self.components["conversation"].process_query(context)
                results.append(result)
                
            elif intent == IntentType.DATABASE_QUERY and context.include_database:
                result = await self.components["database"].process_query(context)
                results.append(result)
                
            elif intent == IntentType.DOCUMENT_QUERY and context.include_documents:
                result = await self.components["documents"].process_query(context)
                results.append(result)
                
            elif intent == IntentType.HYBRID_QUERY:
                if context.include_database:
                    db_result = await self.components["database"].process_query(context)
                    results.append(db_result)
                    
                if context.include_documents:
                    doc_result = await self.components["documents"].process_query(context)
                    results.append(doc_result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in orchestrator: {e}")
            return [QueryResult(
                content=[],
                metadata={},
                source="orchestrator",
                success=False,
                error=str(e)
            )]
    
    async def upload_document(self, file_path: str, file_type: str, description: str = None) -> Dict[str, Any]:
        """Upload document using document component"""
        try:
            return await self.components["documents"].upload_document(file_path, file_type, description)
        except Exception as e:
            logger.error(f"Error uploading document: {e}")
            return {
                "status": "error",
                "message": f"Error uploading document: {str(e)}"
            }

# Pydantic models for API
class QueryRequest(BaseModel):
    question: str
    chat_id: Optional[str] = None
    max_results: Optional[int] = 5
    include_documents: bool = True
    include_database: bool = True

class StreamingQueryRequest(BaseModel):
    question: str
    chat_id: Optional[str] = None
    max_results: Optional[int] = 5
    stream: bool = True
    include_documents: bool = True
    include_database: bool = True

class DocumentUploadResponse(BaseModel):
    filename: str
    status: str
    message: str
    document_id: Optional[str] = None

class QueryResponse(BaseModel):
    question: str
    results: List[Dict[str, Any]]
    success: bool
    error: Optional[str] = None
    sources: List[str]
    intent: str

class HealthResponse(BaseModel):
    status: str
    message: str
    components: Dict[str, bool]

class StreamingChunk(BaseModel):
    type: str
    data: Dict[str, Any]
    timestamp: float

# Initialize FastAPI app
app = FastAPI(
    title="Modular SISL RAG API",
    description="DB-GPT inspired modular RAG API with separate database and document components",
    version="3.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global orchestrator
orchestrator = RAGOrchestrator()

@app.on_event("startup")
async def startup_event():
    """Initialize orchestrator on startup"""
    logger.info("Starting Modular SISL RAG API...")
    await orchestrator.initialize()

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint"""
    return HealthResponse(
        status="healthy",
        message="Modular SISL RAG API is running",
        components={
            "orchestrator": True,
            "database_component": "database" in orchestrator.components,
            "document_component": "documents" in orchestrator.components,
            "conversation_component": "conversation" in orchestrator.components
        }
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        message="Modular SISL RAG API is running",
        components={
            "orchestrator": True,
            "database_component": "database" in orchestrator.components,
            "document_component": "documents" in orchestrator.components,
            "conversation_component": "conversation" in orchestrator.components
        }
    )

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Main query endpoint"""
    try:
        logger.info(f"Received query: {request.question}")
        
        # Create query context
        context = QueryContext(
            question=request.question,
            intent=IntentType.GENERAL_CONVERSATION,  # Will be set by orchestrator
            include_database=request.include_database,
            include_documents=request.include_documents,
            max_results=request.max_results
        )
        
        # Process query
        results = await orchestrator.process_query(context)
        
        # Combine results
        combined_results = []
        sources = []
        success = any(r.success for r in results)
        error = None
        
        for result in results:
            if result.success:
                combined_results.extend(result.content)
                sources.append(result.source)
            else:
                error = result.error
        
        return QueryResponse(
            question=request.question,
            results=combined_results,
            success=success,
            error=error,
            sources=sources,
            intent=context.intent.value
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return QueryResponse(
            question=request.question,
            results=[],
            success=False,
            error=str(e),
            sources=[],
            intent="error"
        )

@app.post("/query/stream")
async def query_stream(request: StreamingQueryRequest) -> StreamingResponse:
    """Streaming query endpoint"""
    
    async def generate_stream() -> AsyncGenerator[str, None]:
        try:
            # Send initial status
            yield f"data: {json.dumps(StreamingChunk(type='status', data={'message': 'Processing query...'}, timestamp=asyncio.get_event_loop().time()).dict())}\n\n"
            
            # Create query context
            context = QueryContext(
                question=request.question,
                intent=IntentType.GENERAL_CONVERSATION,
                include_database=request.include_database,
                include_documents=request.include_documents,
                max_results=request.max_results
            )
            
            # Classify intent
            intent = await orchestrator.classify_intent(request.question)
            context.intent = intent
            
            yield f"data: {json.dumps(StreamingChunk(type='status', data={'message': f'Intent classified as: {intent.value}'}, timestamp=asyncio.get_event_loop().time()).dict())}\n\n"
            
            # Process query
            results = await orchestrator.process_query(context)
            
            # Stream results
            for result in results:
                if result.success:
                    if result.source == "database" and "sql" in result.metadata:
                        yield f"data: {json.dumps(StreamingChunk(type='sql', data={'sql': result.metadata['sql']}, timestamp=asyncio.get_event_loop().time()).dict())}\n\n"
                    
                    yield f"data: {json.dumps(StreamingChunk(type='results', data={'rows': result.content, 'source': result.source}, timestamp=asyncio.get_event_loop().time()).dict())}\n\n"
                else:
                    yield f"data: {json.dumps(StreamingChunk(type='error', data={'error': result.error}, timestamp=asyncio.get_event_loop().time()).dict())}\n\n"
            
            # Send completion
            yield f"data: {json.dumps(StreamingChunk(type='complete', data={'success': True}, timestamp=asyncio.get_event_loop().time()).dict())}\n\n"
            
        except Exception as e:
            logger.error(f"Error in streaming query: {str(e)}")
            yield f"data: {json.dumps(StreamingChunk(type='error', data={'error': str(e)}, timestamp=asyncio.get_event_loop().time()).dict())}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
        }
    )

@app.post("/upload-document")
async def upload_document(
    file: UploadFile = File(...),
    description: Optional[str] = Form(None)
) -> DocumentUploadResponse:
    """Upload and process document"""
    try:
        # Create temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name

        # Determine file type
        file_type = file.content_type or mimetypes.guess_type(file.filename)[0]

        # Process document
        result = await orchestrator.upload_document(tmp_file_path, file_type, description)

        # Clean up temp file
        os.unlink(tmp_file_path)

        return DocumentUploadResponse(
            filename=file.filename,
            status=result["status"],
            message=result["message"],
            document_id=f"doc_{result.get('chunks', 0)}" if result["status"] == "success" else None
        )

    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        return DocumentUploadResponse(
            filename=file.filename,
            status="error",
            message=f"Error processing document: {str(e)}",
            document_id=None
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 