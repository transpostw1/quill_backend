# 🧪 Local Testing Guide for RAG API

This guide will help you test the enhanced RAG API with chat ID-based conversation memory features locally before deployment.

## 📋 Prerequisites

1. Python 3.8+
2. All dependencies installed from `requirements.txt`
3. Ollama running locally (for LLM and embeddings)
4. Qdrant running locally (for vector storage)
5. MSSQL database accessible (for testing database queries)

## 🚀 Running the API Locally

1. **Start the API server:**

   ```bash
   python run_local.py
   ```

2. **The API will be available at:**
   - Base URL: `http://localhost:8000`
   - Health endpoint: `http://localhost:8000/health`
   - Query endpoint: `http://localhost:8000/query`
   - Clear memory endpoint: `http://localhost:8000/clear_memory`

## 🧪 Testing the API

### Option 1: Using the test script

```bash
python test_local.py
```

### Option 2: Manual testing with curl

```bash
# Health check
curl http://localhost:8000/health

# Simple greeting (should be handled without RAG)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Hi there!", "chat_id": "test-session-123"}'

# Database query (should go through RAG pipeline)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How many customers do we have?", "chat_id": "test-session-123"}'

# Conversation context test (using same chat_id to maintain context)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What was my previous question?", "chat_id": "test-session-123"}'

# Clear conversation memory for a chat ID
curl -X POST "http://localhost:8000/clear_memory?chat_id=test-session-123"
```

### Option 3: Using PowerShell (Windows)

```powershell
# Health check
Invoke-RestMethod -Uri "http://localhost:8000/health" -Method GET

# Simple greeting with chat_id
Invoke-RestMethod -Uri "http://localhost:8000/query" -Method POST -ContentType "application/json" -Body '{"question": "Hi there!", "chat_id": "test-session-123"}'

# Database query with chat_id
Invoke-RestMethod -Uri "http://localhost:8000/query" -Method POST -ContentType "application/json" -Body '{"question": "How many customers do we have?", "chat_id": "test-session-123"}'

# Clear conversation memory for a chat ID
Invoke-RestMethod -Uri "http://localhost:8000/clear_memory?chat_id=test-session-123" -Method POST
```

## 🔍 What to Look For

1. **Intent Classification**: Simple greetings should be handled quickly without RAG processing
2. **Conversation Context**: The API should remember previous interactions within the same chat_id
3. **Database Queries**: Complex questions should go through the full RAG pipeline
4. **Error Handling**: Errors should be properly handled and saved to conversation memory
5. **Chat ID Isolation**: Different chat_id values should have separate conversation contexts
6. **Memory Clearing**: The clear_memory endpoint should properly clear conversation history for a specific chat_id

## 🛑 Stopping the Server

Press `CTRL+C` in the terminal where the server is running to stop it.

## 📝 Notes

- The API uses chat ID-based `ConversationBufferMemory` to maintain separate conversation contexts for each user session
- Conversation history is considered in intent classification
- All interactions are saved to memory for future context within the same chat_id
- Different chat_id values maintain completely separate conversation histories
- The local setup uses the same configuration as the production version
- Memory cleanup is automatically handled to prevent memory leaks
- The `clear_memory` endpoint allows manual clearing of conversation history for a specific chat_id
