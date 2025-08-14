from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, AsyncGenerator, Union
import uvicorn
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
import logging
import threading
import json
import asyncio
import os
import tempfile
import requests
from pathlib import Path
import mimetypes

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Config ---
QDRANT_URL = "http://192.168.0.120:6333/"
COLLECTION_NAME = "schema_vectors"
DOCUMENTS_COLLECTION = "document_vectors"

# OCR API Configuration
OCR_API_URL = "http://192.168.0.120:3001/api/documents"

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
    title="Enhanced SISL RAG API",
    description="DB-GPT inspired RAG API for querying SISL database and documents using natural language",
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
    generated_sql: Optional[str] = None
    results: List[Dict[str, Any]]
    columns: List[str]
    row_count: int
    success: bool
    error: Optional[str] = None
    document_context: Optional[List[Dict[str, Any]]] = None
    context_sources: List[str]

class HealthResponse(BaseModel):
    status: str
    message: str
    components: Dict[str, bool]

class StreamingChunk(BaseModel):
    type: str  # "status", "sql", "results", "documents", "complete", "error"
    data: Dict[str, Any]
    timestamp: float

# Global variables
qdrant = None
document_store = None
embeddings = None
llm = None
text_splitter = None

# Memory manager
class ChatMemoryManager:
    def __init__(self):
        self.memories = {}
        self.lock = threading.Lock()
    
    def get_memory(self, chat_id: str) -> ConversationBufferMemory:
        with self.lock:
            if chat_id not in self.memories:
                self.memories[chat_id] = ConversationBufferMemory(
                    memory_key="chat_history", 
                    return_messages=True
                )
            return self.memories[chat_id]
    
    def clear_memory(self, chat_id: str):
        with self.lock:
            if chat_id in self.memories:
                del self.memories[chat_id]

memory_manager = ChatMemoryManager()

def initialize_components():
    """Initialize all RAG components"""
    global qdrant, document_store, embeddings, llm, text_splitter
    
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
        
        # Initialize database vector store
        qdrant = QdrantVectorStore(
            collection_name=COLLECTION_NAME,
            embedding=embeddings,
            client=client
        )
        
        # Initialize document vector store
        document_store = QdrantVectorStore(
            collection_name=DOCUMENTS_COLLECTION,
            embedding=embeddings,
            client=client
        )

        # Initialize LLM
        llm = ChatOllama(
            model="mistral:latest", 
            temperature=0.1,
            base_url="http://192.168.0.120:11434"
        )
        
        # Initialize text splitter for documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        logger.info("All components initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        return False

def get_document_loader(file_path: str, file_type: str):
    """Get appropriate document loader based on file type"""
    try:
        if file_type.startswith('image/'):
            # Use OCR API for images
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

async def process_document_with_ocr(file_path: str) -> List[Document]:
    """Process document using OCR API"""
    try:
        with open(file_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(OCR_API_URL, files=files)
            
        if response.status_code == 200:
            result = response.json()
            text = result.get('text', '')
            
            # Create document chunks
            chunks = text_splitter.split_text(text)
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

def search_database_schema(query: str, k: int = 5) -> str:
    """Search database schema context"""
    try:
        if qdrant is None:
            return ""
        
        # Search in database schema collection
        docs = qdrant.similarity_search(query, k=k)
        context = "\n".join([doc.page_content for doc in docs])
        return context
    except Exception as e:
        logger.error(f"Error searching database schema: {e}")
        return ""

def search_documents(query: str, k: int = 5) -> List[Document]:
    """Search document context"""
    try:
        if document_store is None:
            return []
        
        # Search in documents collection
        docs = document_store.similarity_search(query, k=k)
        return docs
    except Exception as e:
        logger.error(f"Error searching documents: {e}")
        return []

def generate_sql(context: str, question: str) -> str:
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
        
        chain = prompt | llm
        result = chain.invoke({"context": context, "question": question})
        return result.content.strip()
    except Exception as e:
        logger.error(f"Error generating SQL: {e}")
        return ""

def run_sql_query(sql_query: str) -> tuple[List[str], List[tuple], int]:
    """Execute SQL query on database"""
    try:
        conn = pyodbc.connect(MSSQL_CONN_STR)
        cursor = conn.cursor()
        cursor.execute(sql_query)
        
        # Get column names
        columns = [column[0] for column in cursor.description]
        
        # Get all rows
        rows = cursor.fetchall()
        row_count = len(rows)
        
        cursor.close()
        conn.close()
        
        return columns, rows, row_count
    except Exception as e:
        logger.error(f"Database query error: {e}")
        raise e

def classify_intent(question: str, chat_history: str = "") -> str:
    """Classify user intent"""
    try:
        intent_prompt = f"""
        Classify the following user input as one of these categories:
        
        1. "database_query" - Questions about data, reports, counts, lists that require database access
        2. "document_query" - Questions about documents, files, content, knowledge base
        3. "general_conversation" - Greetings, casual chat, general questions
        4. "hybrid_query" - Questions that might need both database and document information
        
        Examples:
        - "How many customers do we have?" -> database_query
        - "What does the contract say about payment terms?" -> document_query
        - "Show me sales data and related contracts" -> hybrid_query
        - "Hi, how are you?" -> general_conversation
        
        User input: "{question}"
        Chat history: {chat_history}
        
        Respond with ONLY ONE category: database_query, document_query, general_conversation, or hybrid_query
        """
        
        result = llm.invoke(intent_prompt)
        return result.content.strip().lower()
    except Exception as e:
        logger.error(f"Error classifying intent: {e}")
        return "general_conversation"

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    logger.info("Starting Enhanced SISL RAG API...")
    initialize_components()

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint"""
    return HealthResponse(
        status="healthy",
        message="Enhanced SISL RAG API is running",
        components={
            "qdrant": qdrant is not None,
            "document_store": document_store is not None,
            "embeddings": embeddings is not None,
            "llm": llm is not None
        }
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        message="Enhanced SISL RAG API is running",
        components={
            "qdrant": qdrant is not None,
            "document_store": document_store is not None,
            "embeddings": embeddings is not None,
            "llm": llm is not None
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
        documents = []
        
        if file_type and file_type.startswith('image/'):
            # Use OCR API for images
            documents = await process_document_with_ocr(tmp_file_path)
        else:
            # Use LangChain loaders
            loader = get_document_loader(tmp_file_path, file_type)
            if loader:
                raw_docs = loader.load()
                # Split documents into chunks
                for doc in raw_docs:
                    chunks = text_splitter.split_documents([doc])
                    documents.extend(chunks)
        
        if documents:
            # Add to vector store
            document_store.add_documents(documents)
            
            # Clean up temp file
            os.unlink(tmp_file_path)
            
            return DocumentUploadResponse(
                filename=file.filename,
                status="success",
                message=f"Document processed successfully. Added {len(documents)} chunks to knowledge base.",
                document_id=f"doc_{len(documents)}"
            )
        else:
            return DocumentUploadResponse(
                filename=file.filename,
                status="error",
                message="Failed to process document. No content extracted.",
                document_id=None
            )
            
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        return DocumentUploadResponse(
            filename=file.filename,
            status="error",
            message=f"Error processing document: {str(e)}",
            document_id=None
        )

@app.post("/query", response_model=QueryResponse)
async def query_database_and_documents(request: QueryRequest):
    """Main endpoint for hybrid database and document queries"""
    try:
        logger.info(f"Received query: {request.question}")
        
        # Get conversation memory
        chat_id = request.chat_id or "default"
        conversation_memory = memory_manager.get_memory(chat_id)
        memory_vars = conversation_memory.load_memory_variables({})
        chat_history = memory_vars.get("chat_history", [])
        
        # Format chat history
        history_text = ""
        if chat_history:
            history_text = "\nConversation History:\n"
            for msg in chat_history[-6:]:
                if isinstance(msg, HumanMessage):
                    history_text += f"User: {msg.content}\n"
                elif isinstance(msg, AIMessage):
                    history_text += f"Assistant: {msg.content}\n"
        
        # Classify intent
        intent = classify_intent(request.question, history_text)
        logger.info(f"Intent classified as: {intent}")
        
        results = []
        columns = []
        row_count = 0
        generated_sql = None
        document_context = None
        context_sources = []
        
        # Handle different intents
        if intent == "general_conversation":
            # Handle general conversation
            conversation_prompt = f"""
            You are a helpful assistant for a database and document querying system. 
            Respond appropriately to this user input: "{request.question}"
            Keep your response concise and friendly.
            {history_text}
            """
            
            conversation_result = llm.invoke(conversation_prompt)
            response_text = conversation_result.content.strip()
            
            results = [{"response": response_text}]
            columns = ["response"]
            row_count = 1
            
            # Save to memory
            conversation_memory.save_context(
                {"input": request.question}, 
                {"output": response_text}
            )
            
        elif intent in ["database_query", "hybrid_query"] and request.include_database:
            # Handle database query
            context = search_database_schema(request.question, request.max_results)
            if context:
                generated_sql = generate_sql(context, request.question)
                if generated_sql:
                    columns, rows, row_count = run_sql_query(generated_sql)
                    
                    # Convert rows to list of dictionaries
                    for row in rows:
                        row_dict = {}
                        for i, value in enumerate(row):
                            if hasattr(value, 'isoformat'):
                                row_dict[columns[i]] = value.isoformat()
                            else:
                                row_dict[columns[i]] = str(value) if value is not None else None
                        results.append(row_dict)
                    
                    context_sources.append("database")
            
        if intent in ["document_query", "hybrid_query"] and request.include_documents:
            # Handle document query
            doc_docs = search_documents(request.question, request.max_results)
            if doc_docs:
                document_context = []
                for doc in doc_docs:
                    document_context.append({
                        "content": doc.page_content,
                        "source": doc.metadata.get("source", "unknown"),
                        "chunk_id": doc.metadata.get("chunk_id", 0)
                    })
                context_sources.append("documents")
        
        # Generate final response if needed
        if not results and not document_context:
            # No results found, provide helpful response
            results = [{"response": "I couldn't find relevant information for your query. Please try rephrasing or ask about specific data or documents."}]
            columns = ["response"]
            row_count = 1
        
        return QueryResponse(
            question=request.question,
            generated_sql=generated_sql,
            results=results,
            columns=columns,
            row_count=row_count,
            success=True,
            document_context=document_context,
            context_sources=context_sources
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return QueryResponse(
            question=request.question,
            generated_sql=None,
            results=[],
            columns=[],
            row_count=0,
            success=False,
            error=str(e),
            document_context=None,
            context_sources=[]
        )

@app.post("/query/stream")
async def query_database_and_documents_stream(request: StreamingQueryRequest) -> StreamingResponse:
    """Streaming endpoint for hybrid queries"""
    
    async def generate_stream() -> AsyncGenerator[str, None]:
        try:
            # Send initial status
            yield f"data: {json.dumps(StreamingChunk(type='status', data={'message': 'Processing query...'}, timestamp=asyncio.get_event_loop().time()).dict())}\n\n"
            
            # Intent classification
            intent = classify_intent(request.question)
            yield f"data: {json.dumps(StreamingChunk(type='status', data={'message': f'Intent classified as: {intent}'}, timestamp=asyncio.get_event_loop().time()).dict())}\n\n"
            
            if intent == "general_conversation":
                # Handle conversation
                conversation_result = llm.invoke(f"Respond to: {request.question}")
                yield f"data: {json.dumps(StreamingChunk(type='results', data={'response': conversation_result.content.strip(), 'is_conversation': True}, timestamp=asyncio.get_event_loop().time()).dict())}\n\n"
                yield f"data: {json.dumps(StreamingChunk(type='complete', data={'success': True, 'row_count': 1}, timestamp=asyncio.get_event_loop().time()).dict())}\n\n"
                return
            
            # Process database query if needed
            if intent in ["database_query", "hybrid_query"] and request.include_database:
                yield f"data: {json.dumps(StreamingChunk(type='status', data={'message': 'Searching database schema...'}, timestamp=asyncio.get_event_loop().time()).dict())}\n\n"
                
                context = search_database_schema(request.question, request.max_results)
                if context:
                    yield f"data: {json.dumps(StreamingChunk(type='status', data={'message': 'Generating SQL query...'}, timestamp=asyncio.get_event_loop().time()).dict())}\n\n"
                    
                    generated_sql = generate_sql(context, request.question)
                    if generated_sql:
                        yield f"data: {json.dumps(StreamingChunk(type='sql', data={'sql': generated_sql}, timestamp=asyncio.get_event_loop().time()).dict())}\n\n"
                        
                        yield f"data: {json.dumps(StreamingChunk(type='status', data={'message': 'Executing database query...'}, timestamp=asyncio.get_event_loop().time()).dict())}\n\n"
                        
                        columns, rows, row_count = run_sql_query(generated_sql)
                        
                        # Stream database results
                        results = []
                        for row in rows:
                            row_dict = {}
                            for i, value in enumerate(row):
                                if hasattr(value, 'isoformat'):
                                    row_dict[columns[i]] = value.isoformat()
                                else:
                                    row_dict[columns[i]] = str(value) if value is not None else None
                            results.append(row_dict)
                        
                        # Send results in chunks
                        chunk_size = 3
                        for i in range(0, len(results), chunk_size):
                            chunk = results[i:i + chunk_size]
                            yield f"data: {json.dumps(StreamingChunk(type='results', data={'rows': chunk, 'partial': i + chunk_size < len(results), 'source': 'database'}, timestamp=asyncio.get_event_loop().time()).dict())}\n\n"
                            await asyncio.sleep(0.1)
            
            # Process document query if needed
            if intent in ["document_query", "hybrid_query"] and request.include_documents:
                yield f"data: {json.dumps(StreamingChunk(type='status', data={'message': 'Searching documents...'}, timestamp=asyncio.get_event_loop().time()).dict())}\n\n"
                
                doc_docs = search_documents(request.question, request.max_results)
                if doc_docs:
                    document_context = []
                    for doc in doc_docs:
                        document_context.append({
                            "content": doc.page_content,
                            "source": doc.metadata.get("source", "unknown"),
                            "chunk_id": doc.metadata.get("chunk_id", 0)
                        })
                    
                    yield f"data: {json.dumps(StreamingChunk(type='documents', data={'documents': document_context, 'source': 'documents'}, timestamp=asyncio.get_event_loop().time()).dict())}\n\n"
            
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

@app.get("/documents")
async def list_documents():
    """List all documents in the knowledge base"""
    try:
        # This would require additional implementation to track documents
        # For now, return a placeholder
        return {
            "documents": [],
            "total": 0,
            "message": "Document listing not yet implemented"
        }
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document from the knowledge base"""
    try:
        # This would require additional implementation
        return {"message": f"Document {document_id} deletion not yet implemented"}
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 