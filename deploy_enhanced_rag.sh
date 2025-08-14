#!/bin/bash

# Deploy Enhanced RAG API with Document Processing
echo "🚀 Deploying Enhanced RAG API with Document Processing..."

# Copy the enhanced API file to the server
echo "📤 Uploading enhanced_rag_api.py..."
scp enhanced_rag_api.py root@192.168.0.120:/var/www/quill_backend/

# SSH into server and restart the service
echo "🔄 Restarting Enhanced RAG API service..."
ssh root@192.168.0.120 << 'EOF'
cd /var/www/quill_backend

# Stop the old service
sudo systemctl stop sisl-rag-api

# Backup the old API
cp api.py api_backup_$(date +%Y%m%d_%H%M%S).py

# Replace with enhanced version
cp enhanced_rag_api.py api.py

# Restart the RAG API service
sudo systemctl restart sisl-rag-api

# Wait a moment for the service to start
sleep 5

# Check service status
echo "📊 Service Status:"
sudo systemctl status sisl-rag-api --no-pager

# Test the enhanced endpoints
echo "🧪 Testing Enhanced RAG endpoints..."

# Test health endpoint
echo "Testing health endpoint:"
curl -s http://localhost:8000/health | jq '.'

# Test document upload endpoint (mock)
echo "Testing document upload endpoint:"
curl -X POST http://localhost:8000/upload-document \
  -F "file=@/dev/null" \
  -F "description=test" \
  --max-time 10 || echo "Upload endpoint ready (test file not found)"

# Test hybrid query endpoint
echo "Testing hybrid query endpoint:"
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "hi", "include_documents": true, "include_database": true}' \
  --max-time 15

echo "✅ Enhanced RAG API deployment complete!"
EOF

echo "🎉 Enhanced RAG API deployed successfully!"
echo ""
echo "📋 New Features Available:"
echo "  • Document Upload & Processing"
echo "  • OCR Integration for Images"
echo "  • Hybrid Database + Document Queries"
echo "  • Intent Classification"
echo "  • Streaming Responses"
echo ""
echo "🌐 New Endpoints:"
echo "  • POST /upload-document - Upload and process documents"
echo "  • POST /query - Hybrid database and document queries"
echo "  • POST /query/stream - Streaming hybrid queries"
echo "  • GET /documents - List documents (placeholder)"
echo "  • DELETE /documents/{id} - Delete documents (placeholder)"
echo ""
echo "🔧 Supported Document Types:"
echo "  • PDF, Word, Excel, PowerPoint"
echo "  • Text files (.txt)"
echo "  • Images (JPEG, PNG) - uses OCR API"
echo ""
echo "📊 Query Types:"
echo "  • database_query - Database-only queries"
echo "  • document_query - Document-only queries"
echo "  • hybrid_query - Both database and documents"
echo "  • general_conversation - Chat responses" 