#!/bin/bash

echo "🚀 Deploying SISL RAG API (Correct Version) to Ubuntu Server..."

# Update system packages
echo "📦 Updating system packages..."
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv unixodbc unixodbc-dev

# Create application directory
APP_DIR="/home/$USER/rag-api"
echo "📁 Creating application directory: $APP_DIR"
mkdir -p $APP_DIR
cd $APP_DIR

# Create virtual environment
echo "🐍 Creating Python virtual environment..."
python3 -m venv rag_env
source rag_env/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install dependencies with specific order to avoid conflicts
echo "📚 Installing Python dependencies..."
pip install pydantic>=2.7.4
pip install fastapi>=0.104.1
pip install uvicorn[standard]>=0.24.0
pip install langchain>=0.3.0
pip install langchain-community>=0.0.20
pip install langchain-ollama>=0.1.0
pip install langchain-qdrant>=0.1.0
pip install pyodbc>=5.0.1
pip install qdrant-client>=1.7.0
pip install requests>=2.31.0
pip install python-multipart>=0.0.6

echo "✅ Dependencies installed successfully!"

# Copy API files (assuming they're uploaded to this directory)
echo "📋 Setting up API files..."
# The API file should be copied here manually or via scp

# Create systemd service
echo "🔧 Creating systemd service..."
sudo tee /etc/systemd/system/sisl-rag-api.service > /dev/null <<EOF
[Unit]
Description=SISL RAG API
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$APP_DIR
Environment=PYTHONPATH=$APP_DIR
ExecStart=$APP_DIR/rag_env/bin/python api.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd and enable service
echo "⚙️ Enabling systemd service..."
sudo systemctl daemon-reload
sudo systemctl enable sisl-rag-api

echo "🎉 Deployment completed!"
echo ""
echo "📋 Next steps:"
echo "1. Upload your api.py to $APP_DIR/"
echo "2. Start the service: sudo systemctl start sisl-rag-api"
echo "3. Check status: sudo systemctl status sisl-rag-api"
echo "4. View logs: sudo journalctl -u sisl-rag-api -f"
echo ""
echo "🔗 Your API will be available at: http://your-server-ip:8000"
echo ""
echo "🧪 Test commands:"
echo "curl http://your-server-ip:8000/health"
echo "curl -X POST http://your-server-ip:8000/query \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -d '{\"question\": \"How many customers do I have?\"}'" 