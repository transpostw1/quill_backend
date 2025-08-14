# Modular RAG Architecture - DB-GPT Inspired

## Overview

This document describes the new modular RAG (Retrieval-Augmented Generation) architecture that separates database and document RAG into distinct components with a middle orchestrator, similar to DB-GPT's approach.

## Architecture Components

### 1. BaseRAGComponent (Abstract Base Class)

- **Purpose**: Defines the interface for all RAG components
- **Key Methods**:
  - `initialize()`: Initialize component resources
  - `process_query()`: Process queries and return results
  - `is_relevant()`: Check if component is relevant for a query

### 2. DatabaseRAGComponent

- **Purpose**: Handles database-specific queries and SQL generation
- **Features**:
  - Schema context retrieval from Qdrant
  - SQL query generation using LLM
  - Database query execution
  - Result formatting and metadata extraction
- **Vector Store**: Uses `schema_vectors` collection
- **Intent Detection**: Determines if query requires database access

### 3. DocumentRAGComponent

- **Purpose**: Handles document queries and file processing
- **Features**:
  - Document upload and processing
  - OCR integration for image files
  - Multiple file format support (PDF, Excel, PowerPoint, etc.)
  - Document chunking and vectorization
  - Document similarity search
- **Vector Store**: Uses `document_vectors` collection
- **OCR API**: Integrates with external OCR service

### 4. ConversationComponent

- **Purpose**: Handles general conversation and greetings
- **Features**:
  - Natural language responses
  - Conversation memory management
  - Intent classification for casual chat
- **Temperature**: Higher temperature (0.7) for more natural responses

### 5. RAGOrchestrator

- **Purpose**: Main orchestrator that coordinates all components
- **Features**:
  - Intent classification and routing
  - Component initialization and management
  - Query processing coordination
  - Result aggregation and formatting
  - Memory management

## Intent Classification System

### Intent Types

1. **DATABASE_QUERY**: Questions requiring database access
2. **DOCUMENT_QUERY**: Questions about documents/knowledge base
3. **GENERAL_CONVERSATION**: Greetings and casual chat
4. **HYBRID_QUERY**: Questions requiring both database and documents

### Classification Logic

- Each component implements `is_relevant()` method
- Orchestrator queries all components for relevance
- Intent determined based on relevant components:
  - Only conversation → GENERAL_CONVERSATION
  - Database + Documents → HYBRID_QUERY
  - Only database → DATABASE_QUERY
  - Only documents → DOCUMENT_QUERY

## API Endpoints

### Core Endpoints

- `GET /`: Root endpoint with health status
- `GET /health`: Health check with component status
- `POST /query`: Main query endpoint
- `POST /query/stream`: Streaming query endpoint
- `POST /upload-document`: Document upload endpoint

### Request Models

```python
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
```

### Response Models

```python
class QueryResponse(BaseModel):
    question: str
    results: List[Dict[str, Any]]
    success: bool
    error: Optional[str] = None
    sources: List[str]
    intent: str

class DocumentUploadResponse(BaseModel):
    filename: str
    status: str
    message: str
    document_id: Optional[str] = None
```

## Configuration

### Environment Variables

```python
QDRANT_URL = "http://192.168.0.120:6333/"
COLLECTION_NAME = "schema_vectors"
DOCUMENTS_COLLECTION = "document_vectors"
OCR_API_URL = "http://192.168.0.120:3001/api/documents"
MSSQL_CONN_STR = "DRIVER={ODBC Driver 17 for SQL Server};..."
```

### Component Initialization

Each component initializes:

- **Embeddings**: OllamaEmbeddings with `nomic-embed-text`
- **LLM**: ChatOllama with `mistral:latest`
- **Vector Store**: QdrantVectorStore with appropriate collection
- **Text Splitter**: RecursiveCharacterTextSplitter (document component)

## Document Processing Pipeline

### Supported File Types

- **Images**: OCR processing via external API
- **PDFs**: UnstructuredPDFLoader
- **Excel**: UnstructuredExcelLoader
- **PowerPoint**: UnstructuredPowerPointLoader
- **Text**: TextLoader
- **Other**: UnstructuredFileLoader

### Processing Steps

1. **File Upload**: Temporary file creation
2. **Type Detection**: MIME type identification
3. **Loader Selection**: Appropriate LangChain loader
4. **Content Extraction**: Text extraction from file
5. **Chunking**: Text splitting into manageable chunks
6. **Vectorization**: Embedding generation and storage
7. **Metadata**: Source tracking and chunk information

## Database Query Pipeline

### Processing Steps

1. **Intent Detection**: Determine if database query needed
2. **Schema Search**: Retrieve relevant schema context
3. **SQL Generation**: LLM-based SQL query generation
4. **Query Execution**: Database query execution
5. **Result Formatting**: Data formatting and metadata extraction

## Streaming Response Format

### Chunk Types

- **status**: Processing status updates
- **sql**: Generated SQL queries
- **results**: Query results (database or documents)
- **documents**: Document search results
- **complete**: Processing completion
- **error**: Error messages

### Example Stream

```
data: {"type": "status", "data": {"message": "Processing query..."}, "timestamp": 1234567890.123}
data: {"type": "status", "data": {"message": "Intent classified as: database_query"}, "timestamp": 1234567890.124}
data: {"type": "sql", "data": {"sql": "SELECT * FROM customers"}, "timestamp": 1234567890.125}
data: {"type": "results", "data": {"rows": [...], "source": "database"}, "timestamp": 1234567890.126}
data: {"type": "complete", "data": {"success": true}, "timestamp": 1234567890.127}
```

## Deployment

### Prerequisites

- Python 3.8+
- Required packages (see requirements.txt)
- Access to Ollama server
- Access to Qdrant server
- Access to MSSQL database
- OCR API service (optional)

### Deployment Scripts

- `deploy_modular_rag.sh`: Bash deployment script
- `deploy_modular_rag.ps1`: PowerShell deployment script

### Deployment Steps

1. **Backup**: Create backup of current API
2. **Stop Service**: Stop current RAG service
3. **Deploy**: Copy new modular architecture
4. **Start Service**: Start updated service
5. **Test**: Verify all endpoints work correctly

## Benefits of Modular Architecture

### 1. Separation of Concerns

- Database and document processing are isolated
- Each component has specific responsibilities
- Easier to maintain and debug

### 2. Scalability

- Components can be scaled independently
- New components can be added easily
- Resource allocation can be optimized

### 3. Flexibility

- Components can be enabled/disabled
- Different components can use different models
- Easy to swap implementations

### 4. Testability

- Each component can be tested independently
- Mock components for testing
- Better error isolation

### 5. Extensibility

- New RAG components can be added
- Different vector stores can be used
- Multiple LLM providers supported

## Error Handling

### Component-Level Errors

- Each component handles its own errors
- Errors are wrapped in QueryResult objects
- Detailed error messages for debugging

### Orchestrator-Level Errors

- Graceful degradation when components fail
- Fallback to conversation component
- Error aggregation and reporting

### API-Level Errors

- HTTP status codes for different error types
- Structured error responses
- Detailed logging for troubleshooting

## Monitoring and Logging

### Logging Levels

- **INFO**: Normal operation messages
- **ERROR**: Error conditions and exceptions
- **DEBUG**: Detailed debugging information

### Health Checks

- Component status monitoring
- Service availability checks
- Resource usage tracking

### Metrics

- Query processing times
- Component usage statistics
- Error rates and types

## Future Enhancements

### Planned Features

1. **Component Configuration**: Dynamic component configuration
2. **Model Switching**: Easy model switching per component
3. **Caching**: Result caching for performance
4. **Rate Limiting**: API rate limiting
5. **Authentication**: API authentication and authorization
6. **Metrics Dashboard**: Real-time monitoring dashboard

### Potential Components

1. **WebRAGComponent**: Web search integration
2. **CodeRAGComponent**: Code analysis and generation
3. **ImageRAGComponent**: Image analysis and description
4. **AudioRAGComponent**: Speech-to-text and audio processing

## Troubleshooting

### Common Issues

1. **Component Initialization Failures**: Check service connectivity
2. **Query Processing Errors**: Verify component dependencies
3. **Document Upload Issues**: Check file permissions and OCR API
4. **Database Connection Errors**: Verify MSSQL connectivity

### Debug Commands

```bash
# Check service status
sudo systemctl status sisl-rag-api

# View logs
sudo journalctl -u sisl-rag-api -f

# Test health endpoint
curl http://localhost:8000/health

# Test query endpoint
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "test"}'
```

## Conclusion

The modular RAG architecture provides a robust, scalable, and maintainable solution for hybrid database and document querying. By separating concerns and using a central orchestrator, the system can handle complex queries while remaining easy to extend and modify.
