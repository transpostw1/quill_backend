# Deploy PPR-enabled RAG API to production server (PowerShell)

Write-Host "🚀 Deploying PPR-enabled RAG API..." -ForegroundColor Green

# Copy the updated API file to the server
Write-Host "📤 Uploading updated api.py..." -ForegroundColor Yellow
scp api.py root@192.168.0.120:/var/www/quill_backend/

# SSH into server and restart the service
Write-Host "🔄 Restarting RAG API service..." -ForegroundColor Yellow
ssh root@192.168.0.120 @"
cd /var/www/quill_backend

# Restart the RAG API service
sudo systemctl restart sisl-rag-api

# Wait a moment for the service to start
sleep 3

# Check service status
echo "📊 Service Status:"
sudo systemctl status sisl-rag-api --no-pager

# Test the new PPR endpoint
echo "🧪 Testing PPR endpoint..."
curl -X POST http://localhost:8000/query/ppr \
  -H "Content-Type: application/json" \
  -d '{"question": "hi"}' \
  --max-time 10

echo "✅ PPR API deployment complete!"
"@

Write-Host "🎉 PPR-enabled RAG API deployed successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "📋 New endpoints available:" -ForegroundColor Cyan
Write-Host "  • POST /query/stream - Full streaming endpoint" -ForegroundColor White
Write-Host "  • POST /query/ppr    - PPR-optimized streaming endpoint" -ForegroundColor White
Write-Host ""
Write-Host "🌐 Test URLs:" -ForegroundColor Cyan
Write-Host "  • http://192.168.0.120:8000/query/ppr" -ForegroundColor White
Write-Host "  • http://192.168.0.120:8000/query/stream" -ForegroundColor White 