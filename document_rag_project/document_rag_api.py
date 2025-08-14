from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, AsyncGenerator, Union
import uvicorn
from langchain_qdrant import QdrantVectorStore
from langchain_ollama import OllamaEmbeddings, ChatOllama
import qdrant_client
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
    UnstructuredPowerPointLoader,
    CSVLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredMarkdownLoader
)
import logging
import json
import asyncio
import os
import tempfile
import hashlib
from pathlib import Path
import mimetypes
from datetime import datetime
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
QDRANT_URL = "http://192.168.0.120:6333/"
DOCUMENTS_COLLECTION = "document_vectors"
OLLAMA_BASE_URL = "http://192.168.0.120:11434"
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2:3b"

# Document processing settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB

# Initialize FastAPI app
app = FastAPI(
    title="Document RAG API",
    description="Specialized RAG API for processing and searching large documents using Ollama and Qdrant",
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

# Global components
embeddings = None
vector_store = None
llm = None
text_splitter = None

# Pydantic models
class DocumentUploadRequest(BaseModel):
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    status: str
    message: str
    processing_status: str
    chunk_count: Optional[int] = None
    file_size: int
    upload_time: str

class SearchRequest(BaseModel):
    query: str
    max_results: int = 10
    similarity_threshold: float = 0.7
    filters: Optional[Dict[str, Any]] = None
    include_metadata: bool = True

class SearchResponse(BaseModel):
    query: str
    results: List[Dict[str, Any]]
    total_results: int
    search_time: float
    metadata: Dict[str, Any]

class ChatRequest(BaseModel):
    question: str
    context_documents: Optional[List[str]] = None
    chat_history: Optional[List[Dict[str, str]]] = None
    max_context_length: int = 4000

class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    processing_time: float

class DocumentInfo(BaseModel):
    document_id: str
    filename: str
    description: Optional[str]
    tags: List[str]
    upload_time: str
    chunk_count: int
    file_size: int
    status: str

class HealthResponse(BaseModel):
    status: str
    message: str
    components: Dict[str, bool]
    stats: Dict[str, Any]

# Document processing functions
def get_document_loader(file_path: str, file_type: str):
    """Get appropriate document loader based on file type"""
    try:
        if file_type == "pdf":
            return PyPDFLoader(file_path)
        elif file_type == "txt":
            return TextLoader(file_path, encoding='utf-8')
        elif file_type == "csv":
            return CSVLoader(file_path)
        elif file_type == "docx":
            return UnstructuredWordDocumentLoader(file_path)
        elif file_type == "md":
            return UnstructuredMarkdownLoader(file_path)
        elif file_type in ["xlsx", "xls"]:
            return UnstructuredExcelLoader(file_path)
        elif file_type in ["pptx", "ppt"]:
            return UnstructuredPowerPointLoader(file_path)
        else:
            return UnstructuredFileLoader(file_path)
    except Exception as e:
        logger.error(f"Error creating loader for {file_path}: {e}")
        return UnstructuredFileLoader(file_path)

def process_document(file_path: str, document_id: str, metadata: Dict[str, Any]) -> List[Document]:
    """Process document and return chunks"""
    try:
        file_extension = Path(file_path).suffix.lower().lstrip('.')
        loader = get_document_loader(file_path, file_extension)
        documents = loader.load()
        
        # Add metadata to each document
        for doc in documents:
            doc.metadata.update(metadata)
            doc.metadata['document_id'] = document_id
            doc.metadata['chunk_id'] = str(uuid.uuid4())
            doc.metadata['processed_at'] = datetime.now().isoformat()
        
        # Split documents into chunks
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Processed {len(chunks)} chunks from {file_path}")
        
        return chunks
    except Exception as e:
        logger.error(f"Error processing document {file_path}: {e}")
        raise

def initialize_components():
    """Initialize all components"""
    global embeddings, vector_store, llm, text_splitter
    
    try:
        # Initialize embeddings
        embeddings = OllamaEmbeddings(
            model=EMBEDDING_MODEL,
            base_url=OLLAMA_BASE_URL
        )
        
        # Initialize text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Initialize vector store
        vector_store = QdrantVectorStore(
            client=qdrant_client.QdrantClient(url=QDRANT_URL),
            collection_name=DOCUMENTS_COLLECTION,
            embeddings=embeddings
        )
        
        # Initialize LLM
        llm = ChatOllama(
            model=LLM_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.1
        )
        
        logger.info("All components initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Error initializing components: {e}")
        return False

# Background task for document processing
async def process_document_background(file_path: str, document_id: str, metadata: Dict[str, Any]):
    """Background task to process document"""
    try:
        chunks = process_document(file_path, document_id, metadata)
        
        # Add chunks to vector store
        vector_store.add_documents(chunks)
        
        # Update metadata with chunk count
        metadata['chunk_count'] = len(chunks)
        metadata['status'] = 'processed'
        
        logger.info(f"Document {document_id} processed successfully with {len(chunks)} chunks")
        
    except Exception as e:
        logger.error(f"Error in background processing for {document_id}: {e}")
        metadata['status'] = 'error'
        metadata['error'] = str(e)

# API Endpoints
@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    success = initialize_components()
    if not success:
        logger.error("Failed to initialize components")

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint with health check"""
    try:
        # Check component health
        components = {
            "qdrant": vector_store is not None,
            "embeddings": embeddings is not None,
            "llm": llm is not None
        }
        
        # Get collection stats
        stats = {}
        if vector_store:
            try:
                collection_info = vector_store.client.get_collection(DOCUMENTS_COLLECTION)
                stats = {
                    "total_vectors": collection_info.points_count,
                    "collection_name": DOCUMENTS_COLLECTION
                }
            except:
                stats = {"error": "Could not retrieve collection stats"}
        
        return HealthResponse(
            status="healthy" if all(components.values()) else "degraded",
            message="Document RAG API is running",
            components=components,
            stats=stats
        )
    except Exception as e:
        return HealthResponse(
            status="unhealthy",
            message=f"Error: {str(e)}",
            components={},
            stats={}
        )

@app.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    description: Optional[str] = Form(None),
    tags: Optional[str] = Form(None),
    metadata: Optional[str] = Form(None)
):
    """Upload and process a document"""
    try:
        # Validate file size
        if file.size > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail=f"File too large. Maximum size is {MAX_FILE_SIZE} bytes")
        
        # Generate document ID
        document_id = str(uuid.uuid4())
        
        # Parse optional parameters
        tag_list = json.loads(tags) if tags else []
        metadata_dict = json.loads(metadata) if metadata else {}
        
        # Create metadata
        doc_metadata = {
            "filename": file.filename,
            "description": description,
            "tags": tag_list,
            "upload_time": datetime.now().isoformat(),
            "file_size": file.size,
            "content_type": file.content_type,
            "status": "processing",
            **metadata_dict
        }
        
        # Save file temporarily
        temp_dir = Path("temp_documents")
        temp_dir.mkdir(exist_ok=True)
        
        file_path = temp_dir / f"{document_id}_{file.filename}"
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Start background processing
        background_tasks.add_task(
            process_document_background,
            str(file_path),
            document_id,
            doc_metadata
        )
        
        return DocumentUploadResponse(
            document_id=document_id,
            filename=file.filename,
            status="uploaded",
            message="Document uploaded successfully. Processing in background.",
            processing_status="processing",
            file_size=file.size,
            upload_time=doc_metadata["upload_time"]
        )
        
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """Search documents using semantic similarity"""
    try:
        start_time = datetime.now()
        
        # Perform similarity search
        results = vector_store.similarity_search_with_score(
            request.query,
            k=request.max_results,
            filter=request.filters
        )
        
        # Format results
        formatted_results = []
        for doc, score in results:
            result = {
                "content": doc.page_content,
                "score": float(score),
                "document_id": doc.metadata.get("document_id"),
                "chunk_id": doc.metadata.get("chunk_id"),
                "filename": doc.metadata.get("filename")
            }
            
            if request.include_metadata:
                result["metadata"] = doc.metadata
            
            # Apply similarity threshold
            if score <= request.similarity_threshold:
                formatted_results.append(result)
        
        search_time = (datetime.now() - start_time).total_seconds()
        
        return SearchResponse(
            query=request.query,
            results=formatted_results,
            total_results=len(formatted_results),
            search_time=search_time,
            metadata={
                "similarity_threshold": request.similarity_threshold,
                "max_results": request.max_results
            }
        )
        
    except Exception as e:
        logger.error(f"Error searching documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
async def chat_with_documents(request: ChatRequest):
    """Chat with documents using RAG"""
    try:
        start_time = datetime.now()
        
        # Search for relevant documents
        search_results = vector_store.similarity_search(
            request.question,
            k=5
        )
        
        # Build context from documents
        context = "\n\n".join([doc.page_content for doc in search_results])
        
        # Truncate context if too long
        if len(context) > request.max_context_length:
            context = context[:request.max_context_length] + "..."
        
        # Build chat history
        messages = []
        if request.chat_history:
            for msg in request.chat_history:
                if msg.get("role") == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg.get("role") == "assistant":
                    messages.append(AIMessage(content=msg["content"]))
        
        # Add current question
        messages.append(HumanMessage(content=request.question))
        
        # Create prompt template
        prompt_template = ChatPromptTemplate.from_template("""
        You are a helpful assistant that answers questions based on the provided document context.
        
        Context from documents:
        {context}
        
        Question: {question}
        
        Please answer the question based on the context provided. If the context doesn't contain enough information to answer the question, say so. Be concise and accurate.
        """)
        
        # Generate response
        chain = prompt_template | llm
        response = chain.invoke({
            "context": context,
            "question": request.question
        })
        
        # Prepare sources
        sources = []
        for doc in search_results:
            sources.append({
                "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "filename": doc.metadata.get("filename"),
                "document_id": doc.metadata.get("document_id")
            })
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return ChatResponse(
            answer=response.content,
            sources=sources,
            confidence=0.8,  # Placeholder
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error in chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents", response_model=List[DocumentInfo])
async def list_documents():
    """List all uploaded documents"""
    try:
        # This would typically query a database for document metadata
        # For now, return empty list as we don't have persistent metadata storage
        return []
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document and its vectors"""
    try:
        # Delete vectors from Qdrant
        vector_store.client.delete(
            collection_name=DOCUMENTS_COLLECTION,
            points_selector={"filter": {"must": [{"key": "document_id", "match": {"value": document_id}}]}}
        )
        
        return {"message": f"Document {document_id} deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 