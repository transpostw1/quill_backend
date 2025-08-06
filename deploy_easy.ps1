# Configuration
$SERVER_IP = "192.168.0.120"
$SERVER_USER = "root"
$SERVER_DIR = "/home/root/rag-api"
$LOCAL_API_FILE = "api.py"

Write-Host "🚀 EASY DEPLOYMENT: SISL RAG API (Correct Version)" -ForegroundColor Green
Write-Host "==================================================" -ForegroundColor Green

# Check if api.py exists locally
if (-not (Test-Path $LOCAL_API_FILE)) {
    Write-Host "❌ Error: $LOCAL_API_FILE not found in current directory" -ForegroundColor Red
    Write-Host "Please make sure api.py is in the same directory as this script" -ForegroundColor Red
    exit 1
}

Write-Host "📤 Uploading files to server..." -ForegroundColor Yellow
scp $LOCAL_API_FILE "${SERVER_USER}@${SERVER_IP}:${SERVER_DIR}/"
scp deploy_server.sh "${SERVER_USER}@${SERVER_IP}:${SERVER_DIR}/"

Write-Host "🔧 Running deployment on server..." -ForegroundColor Yellow
ssh "${SERVER_USER}@${SERVER_IP}" @"
cd /home/root/rag-api
chmod +x deploy_server.sh
./deploy_server.sh
"@

Write-Host "🚀 Starting the service..." -ForegroundColor Yellow
ssh "${SERVER_USER}@${SERVER_IP}" "sudo systemctl start sisl-rag-api"

Write-Host "📊 Checking service status..." -ForegroundColor Yellow
ssh "${SERVER_USER}@${SERVER_IP}" "sudo systemctl status sisl-rag-api"

Write-Host ""
Write-Host "✅ DEPLOYMENT COMPLETE!" -ForegroundColor Green
Write-Host "==================================================" -ForegroundColor Green
Write-Host "🌐 API URL: http://${SERVER_IP}:8000" -ForegroundColor Cyan
Write-Host ""
Write-Host "🧪 Test the deployment:" -ForegroundColor Yellow
Write-Host "curl http://${SERVER_IP}:8000/health" -ForegroundColor Gray
Write-Host "curl -X POST http://${SERVER_IP}:8000/query -H 'Content-Type: application/json' -d '{\"question\": \"How many customers do I have?\"}'" -ForegroundColor Gray
Write-Host ""
Write-Host "📋 Service management:" -ForegroundColor Yellow
Write-Host "ssh ${SERVER_USER}@${SERVER_IP} 'sudo systemctl status sisl-rag-api'" -ForegroundColor Gray
Write-Host "ssh ${SERVER_USER}@${SERVER_IP} 'sudo journalctl -u sisl-rag-api -f'" -ForegroundColor Gray
Write-Host "ssh ${SERVER_USER}@${SERVER_IP} 'sudo systemctl restart sisl-rag-api'" -ForegroundColor Gray 