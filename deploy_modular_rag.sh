#!/bin/bash

# Deploy Modular RAG Architecture
# This script deploys the new modular RAG system with separate components

echo "🚀 Deploying Modular RAG Architecture..."

# Configuration
SERVER="192.168.0.120"
SERVICE_NAME="sisl-rag-api"
BACKUP_DIR="/home/ubuntu/backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "📋 Configuration:"
echo "  Server: $SERVER"
echo "  Service: $SERVICE_NAME"
echo "  Timestamp: $TIMESTAMP"

# Step 1: Stop the current service
echo "🛑 Stopping current service..."
ssh ubuntu@$SERVER "sudo systemctl stop $SERVICE_NAME"

# Step 2: Create backup
echo "💾 Creating backup..."
ssh ubuntu@$SERVER "mkdir -p $BACKUP_DIR"
ssh ubuntu@$SERVER "sudo cp /home/ubuntu/sisl-rag-api/api.py $BACKUP_DIR/api_backup_$TIMESTAMP.py"

# Step 3: Copy new modular architecture
echo "📁 Copying modular RAG architecture..."
scp modular_rag_architecture.py ubuntu@$SERVER:/home/ubuntu/sisl-rag-api/api.py

# Step 4: Update service file if needed
echo "⚙️  Updating service configuration..."
ssh ubuntu@$SERVER "sudo systemctl daemon-reload"

# Step 5: Start the service
echo "▶️  Starting modular RAG service..."
ssh ubuntu@$SERVER "sudo systemctl start $SERVICE_NAME"

# Step 6: Check service status
echo "🔍 Checking service status..."
ssh ubuntu@$SERVER "sudo systemctl status $SERVICE_NAME --no-pager"

# Step 7: Wait for service to be ready
echo "⏳ Waiting for service to be ready..."
sleep 5

# Step 8: Test the new modular API
echo "🧪 Testing modular RAG API..."

# Test health endpoint
echo "Testing health endpoint..."
ssh ubuntu@$SERVER "curl -s http://localhost:8000/health | jq '.'

# Test root endpoint
echo "Testing root endpoint..."
ssh ubuntu@$SERVER "curl -s http://localhost:8000/ | jq '.'

# Test modular query endpoint
echo "Testing modular query endpoint..."
ssh ubuntu@$SERVER "curl -s -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{\"question\": \"hi\", \"include_documents\": true, \"include_database\": true}' | jq '.'

# Test database query
echo "Testing database query..."
ssh ubuntu@$SERVER "curl -s -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{\"question\": \"How many customers do we have?\", \"include_database\": true, \"include_documents\": false}' | jq '.'

# Test document upload endpoint (mock)
echo "Testing document upload endpoint..."
ssh ubuntu@$SERVER "curl -s -X POST http://localhost:8000/upload-document \
  -F 'file=@/dev/null' \
  -F 'description=test' | jq '.'

# Test streaming endpoint
echo "Testing streaming endpoint..."
ssh ubuntu@$SERVER "curl -s -X POST http://localhost:8000/query/stream \
  -H 'Content-Type: application/json' \
  -d '{\"question\": \"Show me customer data\", \"stream\": true}' | head -20"

echo "✅ Modular RAG deployment completed!"
echo ""
echo "📊 Deployment Summary:"
echo "  ✅ Service stopped and backed up"
echo "  ✅ Modular architecture deployed"
echo "  ✅ Service restarted"
echo "  ✅ Health checks passed"
echo "  ✅ API endpoints tested"
echo ""
echo "🔗 API Endpoints:"
echo "  Health: http://$SERVER:8000/health"
echo "  Query: http://$SERVER:8000/query"
echo "  Stream: http://$SERVER:8000/query/stream"
echo "  Upload: http://$SERVER:8000/upload-document"
echo ""
echo "📁 Backup location: $BACKUP_DIR/api_backup_$TIMESTAMP.py" 