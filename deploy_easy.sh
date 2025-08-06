#!/bin/bash

# Configuration
SERVER_IP="192.168.0.120"
SERVER_USER="root"
SERVER_DIR="/home/root/rag-api"
LOCAL_API_FILE="api.py"

echo "🚀 EASY DEPLOYMENT: SISL RAG API (Correct Version)"
echo "=================================================="

# Check if api.py exists locally
if [ ! -f "$LOCAL_API_FILE" ]; then
    echo "❌ Error: $LOCAL_API_FILE not found in current directory"
    echo "Please make sure api.py is in the same directory as this script"
    exit 1
fi

echo "📤 Uploading files to server..."
scp "$LOCAL_API_FILE" "$SERVER_USER@$SERVER_IP:$SERVER_DIR/"
scp deploy_server.sh "$SERVER_USER@$SERVER_IP:$SERVER_DIR/"

echo "🔧 Running deployment on server..."
ssh "$SERVER_USER@$SERVER_IP" << 'EOF'
cd /home/root/rag-api
chmod +x deploy_server.sh
./deploy_server.sh
EOF

echo "🚀 Starting the service..."
ssh "$SERVER_USER@$SERVER_IP" "sudo systemctl start sisl-rag-api"

echo "📊 Checking service status..."
ssh "$SERVER_USER@$SERVER_IP" "sudo systemctl status sisl-rag-api"

echo ""
echo "✅ DEPLOYMENT COMPLETE!"
echo "=================================================="
echo "🌐 API URL: http://$SERVER_IP:8000"
echo ""
echo "🧪 Test the deployment:"
echo "curl http://$SERVER_IP:8000/health"
echo "curl -X POST http://$SERVER_IP:8000/query \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -d '{\"question\": \"How many customers do I have?\"}'"
echo ""
echo "📋 Service management:"
echo "ssh $SERVER_USER@$SERVER_IP 'sudo systemctl status sisl-rag-api'"
echo "ssh $SERVER_USER@$SERVER_IP 'sudo journalctl -u sisl-rag-api -f'"
echo "ssh $SERVER_USER@$SERVER_IP 'sudo systemctl restart sisl-rag-api'" 