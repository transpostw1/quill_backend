# Unified RAG API

A consolidated document and database RAG system that integrates RAGFlow for document search and database RAG for SQL queries.

## Features

- **Document RAG**: Integration with RAGFlow for document search and retrieval
- **Database RAG**: SQL query generation and execution for business data
- **Intent Classification**: Automatic classification of user queries
- **Response Fusion**: Combining information from multiple sources
- **Streaming Support**: Real-time chat responses
- **Health Monitoring**: Component health checks

## Quick Start

### Prerequisites

- Python 3.8+
- RAGFlow server running
- MSSQL database
- Ollama server with required models

### Installation

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Set environment variables:

```bash
# RAGFlow Configuration
RAGFLOW_URL=http://192.168.0.120:82
RAGFLOW_API_KEY=your_api_key_here
RAGFLOW_DEFAULT_DATASET_IDS=dataset1,dataset2

# Database Configuration
DATABASE_CONNECTION_STRING="DRIVER={ODBC Driver 17 for SQL Server};SERVER=192.168.1.44,1433;DATABASE=SISL Live;UID=test;PWD=Test@345;TrustServerCertificate=yes;"
COMPANY_PREFIX=SISL$

# LLM Configuration
OLLAMA_BASE=http://192.168.0.120:11434
LLM_MODEL=qwen2.5:7b

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
```

3. Run the API:

```bash
python api.py
```

## API Endpoints

### Chat Endpoints

- `POST /api/v1/chat` - Regular chat completion
- `POST /api/v1/chat/stream` - Streaming chat completion

### Query Endpoints

- `POST /api/v1/query` - Process queries with intent classification

### Utility Endpoints

- `GET /api/v1/health` - Health check
- `GET /api/v1/config` - Configuration information

## Usage Examples

### Python Client

```python
import requests

# Chat completion
response = requests.post("http://localhost:8000/api/v1/chat", json={
    "message": "What documents do you have about customer contracts?",
    "chat_id": "session123"
})
print(response.json())

# Query with intent classification
response = requests.post("http://localhost:8000/api/v1/query", json={
    "question": "Show me recent sales invoices",
    "include_documents": True,
    "include_database": True
})
print(response.json())
```

### JavaScript Client

```javascript
// Chat completion
const response = await fetch("http://localhost:8000/api/v1/chat", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    message: "What documents do you have about customer contracts?",
    chat_id: "session123",
  }),
});
const result = await response.json();
console.log(result);

// Streaming chat
const streamResponse = await fetch("http://localhost:8000/api/v1/chat/stream", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    message: "Tell me about our sales process",
    stream: true,
  }),
});

const reader = streamResponse.body.getReader();
while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  console.log(new TextDecoder().decode(value));
}
```

## Architecture

The system consists of:

1. **RAGFlow Client**: Handles document search and retrieval
2. **Database RAG Client**: Manages SQL generation and execution
3. **Orchestrator**: Coordinates between components and handles intent classification
4. **FastAPI Application**: Provides REST API interface

## Configuration

All configuration is handled through environment variables. See the environment variables section above for available options.

## Health Monitoring

The `/api/v1/health` endpoint provides health status for all components:

```json
{
  "status": "healthy",
  "components": {
    "ragflow": { "status": "healthy" },
    "database": { "status": "healthy" }
  }
}
```

## Error Handling

The API includes comprehensive error handling with appropriate HTTP status codes and error messages. Check the response body for detailed error information.
