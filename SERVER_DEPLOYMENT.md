# 🚀 Ubuntu Server Deployment Guide

## Quick Deployment (5 minutes)

### 1. Upload Files to Server

```bash
# From your local machine
scp production_rag_api.py user@your-server-ip:/home/user/
scp deploy_server.sh user@your-server-ip:/home/user/
```

### 2. Run Deployment Script

```bash
# On your Ubuntu server
cd /home/user
chmod +x deploy_server.sh
./deploy_server.sh
```

### 3. Upload API File

```bash
# Copy the API file to the correct location
cp production_rag_api.py /home/user/rag-api/
```

### 4. Start the Service

```bash
sudo systemctl start sisl-rag-api
sudo systemctl status sisl-rag-api
```

## Manual Installation (if script fails)

### 1. Install System Dependencies

```bash
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv unixodbc unixodbc-dev
```

### 2. Create Application Directory

```bash
mkdir -p /home/$USER/rag-api
cd /home/$USER/rag-api
```

### 3. Create Virtual Environment

```bash
python3 -m venv rag_env
source rag_env/bin/activate
```

### 4. Install Dependencies (One by One)

```bash
pip install --upgrade pip
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
```

### 5. Create Systemd Service

```bash
sudo tee /etc/systemd/system/sisl-rag-api.service > /dev/null <<EOF
[Unit]
Description=SISL RAG API
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=/home/$USER/rag-api
Environment=PYTHONPATH=/home/$USER/rag-api
ExecStart=/home/$USER/rag-api/rag_env/bin/python production_rag_api.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF
```

### 6. Enable and Start Service

```bash
sudo systemctl daemon-reload
sudo systemctl enable sisl-rag-api
sudo systemctl start sisl-rag-api
```

## Service Management

### Check Status

```bash
sudo systemctl status sisl-rag-api
```

### View Logs

```bash
sudo journalctl -u sisl-rag-api -f
```

### Restart Service

```bash
sudo systemctl restart sisl-rag-api
```

### Stop Service

```bash
sudo systemctl stop sisl-rag-api
```

## Testing the API

### Health Check

```bash
curl http://localhost:8002/health
```

### Test Query

```bash
curl -X POST http://localhost:8002/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How many customers do I have?"}'
```

## Troubleshooting

### If dependencies fail to install:

```bash
# Try installing with --no-deps flag
pip install langchain-ollama --no-deps
pip install langchain-qdrant --no-deps
```

### If service fails to start:

```bash
# Check logs
sudo journalctl -u sisl-rag-api -n 50

# Test manually
cd /home/$USER/rag-api
source rag_env/bin/activate
python production_rag_api.py
```

### If port is already in use:

```bash
# Find what's using the port
sudo netstat -tlnp | grep :8002
sudo lsof -i :8002
```

## Firewall Setup (if needed)

```bash
sudo ufw allow 8002
sudo ufw reload
```

## ✅ Your API will be "Always Running"!

The systemd service ensures:

- ✅ Auto-starts when server boots
- ✅ Restarts if the API crashes
- ✅ Logs to system journal
- ✅ Easy management with systemctl commands
