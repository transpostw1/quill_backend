#!/bin/bash

# Deploy PPR-enabled RAG API to production server

echo "🚀 Deploying PPR-enabled RAG API..."

# Copy the updated API file to the server
echo "📤 Uploading updated api.py..."
scp api.py root@192.168.0.120:/var/www/quill_backend/

# SSH into server and restart the service
echo "🔄 Restarting RAG API service..."
ssh root@192.168.0.120 << 'EOF'
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
EOF

echo "🎉 PPR-enabled RAG API deployed successfully!"
echo ""
echo "📋 New endpoints available:"
echo "  • POST /query/stream - Full streaming endpoint"
echo "  • POST /query/ppr    - PPR-optimized streaming endpoint"
echo ""
echo "🌐 Test URLs:"
echo "  • http://192.168.0.120:8000/query/ppr"
echo "  • http://192.168.0.120:8000/query/stream" 