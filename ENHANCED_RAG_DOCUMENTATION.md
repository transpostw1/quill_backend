# Enhanced RAG System Documentation

## 🎯 Overview

This enhanced RAG (Retrieval-Augmented Generation) system is inspired by DB-GPT and provides a comprehensive solution for querying both databases and documents using natural language. It combines the power of LangChain's document processing with your existing database RAG capabilities.

## 🚀 Key Features

### ✅ **Hybrid Query Processing**
- **Database Queries**: Natural language to SQL conversion
- **Document Queries**: Semantic search across uploaded documents
- **Hybrid Queries**: Combine database and document information
- **Intent Classification**: Automatically determines query type

### ✅ **Document Processing Pipeline**
- **Multi-format Support**: PDF, Word, Excel, PowerPoint, Text, Images
- **OCR Integration**: Automatic text extraction from images using your OCR API
- **Chunking**: Intelligent document splitting for better retrieval
- **Vector Storage**: Documents stored in Qdrant for semantic search

### ✅ **Advanced Features**
- **Streaming Responses**: Real-time query processing with PPR support
- **Conversation Memory**: Maintains context across queries
- **Error Handling**: Graceful fallbacks and detailed error messages
- **PPR Support**: Partial Prerendering for better UX

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Query    │───▶│  Intent         │───▶│  Processing     │
│                 │    │  Classification │    │  Pipeline       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Database       │    │  Document       │    │  Response       │
│  Schema Search  │    │  Vector Search  │    │  Generation     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  SQL Generation │    │  OCR API        │    │  Streaming      │
│  & Execution    │    │  Integration    │    │  Response       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📊 Query Types & Intent Classification

### **1. Database Query**
- **Intent**: `database_query`
- **Examples**:
  - "How many customers do I have?"
  - "Show me top customers by sales"
  - "What are my total sales this month?"
- **Processing**: Schema search → SQL generation → Database execution

### **2. Document Query**
- **Intent**: `document_query`
- **Examples**:
  - "What does the contract say about payment terms?"
  - "Show me the invoice details"
  - "Find documents about customer agreements"
- **Processing**: Document vector search → Content retrieval

### **3. Hybrid Query**
- **Intent**: `hybrid_query`
- **Examples**:
  - "Show me sales data and related contracts"
  - "Customer information with their agreements"
  - "Financial reports and supporting documents"
- **Processing**: Both database and document search → Combined results

### **4. General Conversation**
- **Intent**: `general_conversation`
- **Examples**:
  - "Hi, how are you?"
  - "What can you help me with?"
  - "Thank you"
- **Processing**: Conversational response generation

## 🔧 API Endpoints

### **Core Endpoints**

#### `POST /query`
Main endpoint for hybrid queries.

**Request:**
```json
{
  "question": "How many customers do I have?",
  "chat_id": "optional_session_id",
  "max_results": 5,
  "include_documents": true,
  "include_database": true
}
```

**Response:**
```json
{
  "question": "How many customers do I have?",
  "generated_sql": "SELECT COUNT(*) FROM [SISL Live].[dbo].[ssil_UAT$Customer]",
  "results": [{"": "72"}],
  "columns": [""],
  "row_count": 1,
  "success": true,
  "document_context": null,
  "context_sources": ["database"]
}
```

#### `POST /query/stream`
Streaming endpoint for real-time responses.

**Request:**
```json
{
  "question": "Show me sales data and related contracts",
  "stream": true,
  "include_documents": true,
  "include_database": true
}
```

**Streaming Response:**
```
data: {"type": "status", "data": {"message": "Processing query..."}}
data: {"type": "status", "data": {"message": "Intent classified as: hybrid_query"}}
data: {"type": "sql", "data": {"sql": "SELECT ..."}}
data: {"type": "results", "data": {"rows": [...], "partial": true}}
data: {"type": "documents", "data": {"documents": [...], "source": "documents"}}
data: {"type": "complete", "data": {"success": true}}
```

### **Document Management**

#### `POST /upload-document`
Upload and process documents.

**Request:**
```bash
curl -X POST http://192.168.0.120:8000/upload-document \
  -F "file=@document.pdf" \
  -F "description=Contract document"
```

**Response:**
```json
{
  "filename": "document.pdf",
  "status": "success",
  "message": "Document processed successfully. Added 15 chunks to knowledge base.",
  "document_id": "doc_15"
}
```

#### `GET /documents`
List all documents (placeholder).

#### `DELETE /documents/{document_id}`
Delete a document (placeholder).

### **Health & Status**

#### `GET /health`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "message": "Enhanced SISL RAG API is running",
  "components": {
    "qdrant": true,
    "document_store": true,
    "embeddings": true,
    "llm": true
  }
}
```

## 📁 Document Processing

### **Supported Formats**

| Format | Extension | Processing Method |
|--------|-----------|------------------|
| PDF | `.pdf` | UnstructuredPDFLoader |
| Word | `.doc`, `.docx` | UnstructuredFileLoader |
| Excel | `.xlsx`, `.xls` | UnstructuredExcelLoader |
| PowerPoint | `.pptx`, `.ppt` | UnstructuredPowerPointLoader |
| Text | `.txt` | TextLoader |
| Images | `.jpg`, `.jpeg`, `.png` | OCR API |

### **Processing Pipeline**

1. **File Upload**: User uploads document via `/upload-document`
2. **Format Detection**: Automatic MIME type detection
3. **Content Extraction**:
   - **Text/Office files**: LangChain loaders
   - **Images**: OCR API integration
4. **Chunking**: RecursiveCharacterTextSplitter (1000 chars, 200 overlap)
5. **Embedding**: nomic-embed-text model
6. **Storage**: Qdrant vector database

### **OCR Integration**

The system integrates with your OCR API at `http://192.168.0.120:3001/api/documents`:

```python
async def process_document_with_ocr(file_path: str) -> List[Document]:
    with open(file_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(OCR_API_URL, files=files)
    
    if response.status_code == 200:
        result = response.json()
        text = result.get('text', '')
        # Process text into chunks...
```

## 🧠 Intent Classification

The system uses LLM-based intent classification to determine query type:

```python
def classify_intent(question: str, chat_history: str = "") -> str:
    intent_prompt = f"""
    Classify the following user input as one of these categories:
    
    1. "database_query" - Questions about data, reports, counts, lists
    2. "document_query" - Questions about documents, files, content
    3. "general_conversation" - Greetings, casual chat, general questions
    4. "hybrid_query" - Questions that need both database and document info
    
    User input: "{question}"
    Chat history: {chat_history}
    
    Respond with ONLY ONE category.
    """
    
    result = llm.invoke(intent_prompt)
    return result.content.strip().lower()
```

## 💾 Memory Management

The system maintains conversation memory using LangChain's ConversationBufferMemory:

```python
class ChatMemoryManager:
    def __init__(self):
        self.memories = {}
        self.lock = threading.Lock()
    
    def get_memory(self, chat_id: str) -> ConversationBufferMemory:
        # Returns or creates memory for chat session
```

## 🔄 Streaming Architecture

### **Streaming Response Types**

1. **Status Updates**: Real-time processing status
2. **SQL Generation**: Generated SQL queries
3. **Database Results**: Chunked database results
4. **Document Results**: Retrieved document chunks
5. **Completion**: Query completion status
6. **Errors**: Error messages with details

### **PPR Integration**

The system supports Next.js Partial Prerendering:

```jsx
// Static shell (prerendered)
<StaticShell />

// Dynamic content (streams)
<Suspense fallback={<LoadingFallback />}>
  <EnhancedQueryForm />
</Suspense>
```

## 🛠️ Deployment

### **Quick Deployment**

```bash
# Deploy enhanced RAG system
./deploy_enhanced_rag.sh
```

### **Manual Deployment**

```bash
# 1. Upload enhanced API
scp enhanced_rag_api.py root@192.168.0.120:/var/www/quill_backend/

# 2. SSH to server
ssh root@192.168.0.120

# 3. Backup and replace
cd /var/www/quill_backend
cp api.py api_backup_$(date +%Y%m%d_%H%M%S).py
cp enhanced_rag_api.py api.py

# 4. Restart service
sudo systemctl restart sisl-rag-api

# 5. Check status
sudo systemctl status sisl-rag-api
```

## 🧪 Testing

### **Test Document Upload**

```bash
# Test with a PDF file
curl -X POST http://192.168.0.120:8000/upload-document \
  -F "file=@test_document.pdf" \
  -F "description=Test document"
```

### **Test Hybrid Query**

```bash
# Test database + document query
curl -X POST http://192.168.0.120:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Show me sales data and related contracts",
    "include_documents": true,
    "include_database": true
  }'
```

### **Test Streaming**

```bash
# Test streaming endpoint
curl -N -X POST http://192.168.0.120:8000/query/stream \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How many customers do I have?",
    "stream": true
  }'
```

## 📈 Performance Considerations

### **Optimizations**

1. **Chunking Strategy**: 1000 character chunks with 200 overlap
2. **Vector Search**: k=5 for schema, k=5 for documents
3. **Streaming**: Chunked result delivery
4. **Memory Management**: Thread-safe conversation memory
5. **Error Handling**: Graceful fallbacks

### **Resource Usage**

- **Embeddings**: nomic-embed-text model
- **LLM**: mistral:latest for generation
- **Vector DB**: Qdrant with separate collections
- **OCR**: External API integration

## 🔒 Security Considerations

1. **File Upload**: Temporary file handling with cleanup
2. **Input Validation**: Pydantic models for request validation
3. **Error Handling**: No sensitive data in error messages
4. **CORS**: Configured for cross-origin requests

## 🚨 Troubleshooting

### **Common Issues**

1. **Document Upload Fails**
   - Check OCR API availability
   - Verify file format support
   - Check disk space

2. **Query Classification Issues**
   - Review intent classification prompt
   - Check LLM model availability
   - Verify conversation memory

3. **Streaming Not Working**
   - Check service status
   - Verify Ollama models
   - Check network connectivity

### **Debug Commands**

```bash
# Check service status
sudo systemctl status sisl-rag-api

# View logs
sudo journalctl -u sisl-rag-api -f

# Test health endpoint
curl http://192.168.0.120:8000/health

# Check Ollama models
ollama list
```

## 🎉 Summary

This enhanced RAG system provides:

- ✅ **DB-GPT inspired architecture**
- ✅ **Hybrid database + document queries**
- ✅ **OCR integration for images**
- ✅ **Intent classification**
- ✅ **Streaming responses with PPR**
- ✅ **Conversation memory**
- ✅ **Multi-format document support**
- ✅ **Comprehensive error handling**

The system successfully combines the power of LangChain's document processing with your existing database RAG capabilities, creating a comprehensive solution for natural language querying of both structured and unstructured data. 