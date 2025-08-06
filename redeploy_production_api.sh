#!/bin/bash

echo "🚀 REDEPLOYING SISL RAG API - CORRECT VERSION"
echo "=================================================="

# Stop the current service
echo "📋 Stopping current service..."
sudo systemctl stop sisl-rag-api

# Update the service file
echo "📝 Updating service configuration..."
sudo cp sisl-rag-api.service /etc/systemd/system/

# Reload systemd
echo "🔄 Reloading systemd..."
sudo systemctl daemon-reload

# Start the service
echo "🚀 Starting correct RAG API..."
sudo systemctl start sisl-rag-api

# Check status
echo "📊 Checking service status..."
sudo systemctl status sisl-rag-api

echo ""
echo "✅ REDEPLOYMENT COMPLETE!"
echo "=================================================="
echo "🌐 API URL: http://192.168.0.120:8000"
echo "🔍 Health Check: curl http://192.168.0.120:8000/health"
echo "🧪 Test Query: curl -X POST http://192.168.0.120:8000/query -H 'Content-Type: application/json' -d '{\"question\": \"How many customers do I have?\"}'"
echo ""
echo "📋 Service Management:"
echo "   sudo systemctl status sisl-rag-api"
echo "   sudo systemctl restart sisl-rag-api"
echo "   sudo journalctl -u sisl-rag-api -f" 