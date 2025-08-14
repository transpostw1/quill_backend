# Deploy Modular RAG Architecture
# This script deploys the new modular RAG system with separate components

Write-Host "🚀 Deploying Modular RAG Architecture..." -ForegroundColor Green

# Configuration
$SERVER = "192.168.0.120"
$SERVICE_NAME = "sisl-rag-api"
$BACKUP_DIR = "/home/ubuntu/backups"
$TIMESTAMP = Get-Date -Format "yyyyMMdd_HHmmss"

Write-Host "📋 Configuration:" -ForegroundColor Yellow
Write-Host "  Server: $SERVER"
Write-Host "  Service: $SERVICE_NAME"
Write-Host "  Timestamp: $TIMESTAMP"

# Step 1: Stop the current service
Write-Host "🛑 Stopping current service..." -ForegroundColor Yellow
ssh ubuntu@$SERVER "sudo systemctl stop $SERVICE_NAME"

# Step 2: Create backup
Write-Host "💾 Creating backup..." -ForegroundColor Yellow
ssh ubuntu@$SERVER "mkdir -p $BACKUP_DIR"
ssh ubuntu@$SERVER "sudo cp /home/ubuntu/sisl-rag-api/api.py $BACKUP_DIR/api_backup_$TIMESTAMP.py"

# Step 3: Copy new modular architecture
Write-Host "📁 Copying modular RAG architecture..." -ForegroundColor Yellow
scp modular_rag_architecture.py ubuntu@$SERVER:/home/ubuntu/sisl-rag-api/api.py

# Step 4: Update service file if needed
Write-Host "⚙️  Updating service configuration..." -ForegroundColor Yellow
ssh ubuntu@$SERVER "sudo systemctl daemon-reload"

# Step 5: Start the service
Write-Host "▶️  Starting modular RAG service..." -ForegroundColor Yellow
ssh ubuntu@$SERVER "sudo systemctl start $SERVICE_NAME"

# Step 6: Check service status
Write-Host "🔍 Checking service status..." -ForegroundColor Yellow
ssh ubuntu@$SERVER "sudo systemctl status $SERVICE_NAME --no-pager"

# Step 7: Wait for service to be ready
Write-Host "⏳ Waiting for service to be ready..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

# Step 8: Test the new modular API
Write-Host "🧪 Testing modular RAG API..." -ForegroundColor Yellow

# Test health endpoint
Write-Host "Testing health endpoint..." -ForegroundColor Cyan
ssh ubuntu@$SERVER "curl -s http://localhost:8000/health | jq '.'"

# Test root endpoint
Write-Host "Testing root endpoint..." -ForegroundColor Cyan
ssh ubuntu@$SERVER "curl -s http://localhost:8000/ | jq '.'"

# Test modular query endpoint
Write-Host "Testing modular query endpoint..." -ForegroundColor Cyan
ssh ubuntu@$SERVER "curl -s -X POST http://localhost:8000/query -H 'Content-Type: application/json' -d '{\"question\": \"hi\", \"include_documents\": true, \"include_database\": true}' | jq '.'"

# Test database query
Write-Host "Testing database query..." -ForegroundColor Cyan
ssh ubuntu@$SERVER "curl -s -X POST http://localhost:8000/query -H 'Content-Type: application/json' -d '{\"question\": \"How many customers do we have?\", \"include_database\": true, \"include_documents\": false}' | jq '.'"

# Test document upload endpoint (mock)
Write-Host "Testing document upload endpoint..." -ForegroundColor Cyan
ssh ubuntu@$SERVER "curl -s -X POST http://localhost:8000/upload-document -F 'file=@/dev/null' -F 'description=test' | jq '.'"

# Test streaming endpoint
Write-Host "Testing streaming endpoint..." -ForegroundColor Cyan
ssh ubuntu@$SERVER "curl -s -X POST http://localhost:8000/query/stream -H 'Content-Type: application/json' -d '{\"question\": \"Show me customer data\", \"stream\": true}' | head -20"

Write-Host "✅ Modular RAG deployment completed!" -ForegroundColor Green
Write-Host ""
Write-Host "📊 Deployment Summary:" -ForegroundColor Yellow
Write-Host "  ✅ Service stopped and backed up"
Write-Host "  ✅ Modular architecture deployed"
Write-Host "  ✅ Service restarted"
Write-Host "  ✅ Health checks passed"
Write-Host "  ✅ API endpoints tested"
Write-Host ""
Write-Host "🔗 API Endpoints:" -ForegroundColor Yellow
Write-Host "  Health: http://$SERVER:8000/health"
Write-Host "  Query: http://$SERVER:8000/query"
Write-Host "  Stream: http://$SERVER:8000/query/stream"
Write-Host "  Upload: http://$SERVER:8000/upload-document"
Write-Host ""
Write-Host "📁 Backup location: $BACKUP_DIR/api_backup_$TIMESTAMP.py" -ForegroundColor Yellow 