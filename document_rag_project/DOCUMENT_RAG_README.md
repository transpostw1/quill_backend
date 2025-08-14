# 📚 Document RAG System

A powerful Retrieval Augmented Generation (RAG) system for processing, searching, and chatting with large documents using Ollama and Qdrant.

## 🚀 Features

- **Multi-format Document Support**: PDF, TXT, DOCX, MD, CSV, Excel, PowerPoint
- **Semantic Search**: Find relevant content using AI-powered similarity search
- **Chat Interface**: Ask questions and get answers based on your documents
- **Background Processing**: Large documents are processed asynchronously
- **Web Interface**: Beautiful, responsive web UI for easy interaction
- **RESTful API**: Full API for integration with other applications
- **Vector Storage**: Efficient storage and retrieval using Qdrant
- **Metadata Support**: Tag and categorize your documents
- **Health Monitoring**: Built-in health checks and status monitoring

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Interface │    │   REST API      │    │   Document      │
│   (HTML/JS)     │◄──►│   (FastAPI)     │◄──►│   Processing    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │   Ollama        │    │   Qdrant        │
                       │   (LLM/Embed)   │    │   (Vector DB)   │
                       └─────────────────┘    └─────────────────┘
```

## 📋 Prerequisites

- Python 3.8+
- Ollama server running (with models: `nomic-embed-text`, `llama3.2:3b`)
- Qdrant vector database
- Linux/Unix system (for deployment script)

## 🛠️ Installation

### 1. Clone and Setup

```bash
# Clone the repository (if not already done)
git clone <your-repo>
cd lc_rag_old

# Make deployment script executable
chmod +x deploy_document_rag.sh
```

### 2. Configure Services

Update the configuration in `document_rag_api.py`:

```python
# --- Configuration ---
QDRANT_URL = "http://192.168.0.120:6333/"  # Your Qdrant URL
OLLAMA_BASE_URL = "http://192.168.0.120:11434"  # Your Ollama URL
EMBEDDING_MODEL = "nomic-embed-text"  # Your embedding model
LLM_MODEL = "llama3.2:3b"  # Your LLM model
```

### 3. Deploy

```bash
# Run the deployment script
./deploy_document_rag.sh
```

The script will:

- Create a virtual environment
- Install all dependencies
- Set up the Qdrant collection
- Create a systemd service
- Start the API server

## 🚀 Quick Start

### 1. Start the API

```bash
# If using deployment script (already done)
sudo systemctl status document-rag

# Or manually
python document_rag_api.py
```

### 2. Access the Web Interface

Open `document_rag_interface.html` in your browser, or access:

- API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Health Check: http://localhost:8000/

### 3. Upload Documents

Use the web interface or API to upload documents:

```bash
# Using curl
curl -X POST "http://localhost:8000/upload" \
  -F "file=@your_document.pdf" \
  -F "description=My important document" \
  -F "tags=research,ai,important"
```

### 4. Search and Chat

- **Search**: Find relevant content using natural language queries
- **Chat**: Ask questions and get AI-generated answers based on your documents

## 📖 API Reference

### Endpoints

#### Health Check

```http
GET /
```

Returns system health and component status.

#### Upload Document

```http
POST /upload
Content-Type: multipart/form-data

Parameters:
- file: Document file (required)
- description: Document description (optional)
- tags: JSON array of tags (optional)
- metadata: JSON object of metadata (optional)
```

#### Search Documents

```http
POST /search
Content-Type: application/json

{
  "query": "your search query",
  "max_results": 10,
  "similarity_threshold": 0.7,
  "filters": {},
  "include_metadata": true
}
```

#### Chat with Documents

```http
POST /chat
Content-Type: application/json

{
  "question": "your question",
  "chat_history": [],
  "max_context_length": 4000
}
```

#### List Documents

```http
GET /documents
```

Returns list of uploaded documents.

#### Delete Document

```http
DELETE /documents/{document_id}
```

Deletes a document and its vectors.

## 🔧 Configuration

### Document Processing Settings

```python
CHUNK_SIZE = 1000          # Size of text chunks
CHUNK_OVERLAP = 200        # Overlap between chunks
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB max file size
```

### Supported File Types

- **PDF**: `.pdf` (PyPDFLoader, UnstructuredPDFLoader)
- **Text**: `.txt` (TextLoader)
- **Word**: `.docx` (UnstructuredWordDocumentLoader)
- **Markdown**: `.md` (UnstructuredMarkdownLoader)
- **CSV**: `.csv` (CSVLoader)
- **Excel**: `.xlsx`, `.xls` (UnstructuredExcelLoader)
- **PowerPoint**: `.pptx`, `.ppt` (UnstructuredPowerPointLoader)
- **Other**: Any file type (UnstructuredFileLoader)

## 🎯 Usage Examples

### Python Client

```python
from document_rag_client import DocumentRAGClient

# Initialize client
client = DocumentRAGClient("http://localhost:8000")

# Upload document
result = client.upload_document(
    "document.pdf",
    description="Research paper on AI",
    tags=["research", "ai", "machine-learning"]
)

# Search documents
search_results = client.search_documents(
    "What is machine learning?",
    max_results=5
)

# Chat with documents
chat_response = client.chat_with_documents(
    "Explain the main concepts in this document"
)
```

### Web Interface

1. Open `document_rag_interface.html` in your browser
2. Upload documents using the upload form
3. Search for content using the search interface
4. Chat with your documents using the chat interface
5. Manage documents using the document management section

## 🔍 Advanced Features

### Custom Embeddings

You can use different embedding models by changing the configuration:

```python
EMBEDDING_MODEL = "all-minilm"  # Alternative embedding model
```

### Custom LLM Models

Switch to different LLM models:

```python
LLM_MODEL = "llama3.2:8b"  # Larger model for better responses
```

### Filtering and Metadata

Use metadata filters in search:

```python
filters = {
    "must": [
        {"key": "tags", "match": {"value": "research"}},
        {"key": "upload_time", "range": {"gte": "2024-01-01"}}
    ]
}
```

## 🛡️ Security Considerations

- The API runs on localhost by default
- No authentication is implemented (add for production)
- File uploads are validated for size and type
- Temporary files are cleaned up automatically

## 📊 Monitoring and Logs

### Service Management

```bash
# Check service status
sudo systemctl status document-rag

# View logs
sudo journalctl -u document-rag -f

# Restart service
sudo systemctl restart document-rag
```

### Health Monitoring

The API provides health endpoints:

- `GET /`: Overall system health
- Component status (Qdrant, Ollama, embeddings)
- Collection statistics

## 🔧 Troubleshooting

### Common Issues

1. **Qdrant Connection Failed**

   - Check if Qdrant is running: `curl http://your-qdrant-url:6333/collections`
   - Verify network connectivity
   - Check firewall settings

2. **Ollama Connection Failed**

   - Check if Ollama is running: `curl http://your-ollama-url:11434/api/tags`
   - Verify required models are installed: `ollama list`
   - Install missing models: `ollama pull nomic-embed-text`

3. **Document Processing Fails**

   - Check file format support
   - Verify file size limits
   - Check available disk space
   - Review logs for specific errors

4. **API Not Responding**
   - Check if service is running: `sudo systemctl status document-rag`
   - Verify port availability: `netstat -tlnp | grep 8000`
   - Check firewall settings

### Debug Mode

Run the API in debug mode:

```bash
# Stop the service
sudo systemctl stop document-rag

# Run manually with debug logging
python document_rag_api.py --log-level debug
```

## 📈 Performance Optimization

### For Large Document Collections

1. **Increase Chunk Size**: Adjust `CHUNK_SIZE` for better context
2. **Optimize Embeddings**: Use faster embedding models
3. **Batch Processing**: Process multiple documents in batches
4. **Caching**: Implement response caching for frequent queries

### Scaling Considerations

- Use multiple Qdrant instances for high availability
- Implement load balancing for multiple API instances
- Use Redis for session management and caching
- Consider using GPU-accelerated models

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:

1. Check the troubleshooting section
2. Review the logs: `sudo journalctl -u document-rag -f`
3. Open an issue on GitHub
4. Check the API documentation at `http://localhost:8000/docs`

---

**Happy Document Processing! 📚✨**
