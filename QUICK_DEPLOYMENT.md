# Quick Server Deployment Guide

## Prerequisites

1. **Server Requirements:**

   - Ubuntu 20.04+ or CentOS 8+
   - Python 3.9+
   - 4GB+ RAM
   - 20GB+ disk space

2. **External Services (Already Running):**
   - Qdrant: 192.168.0.120:6333
   - Ollama: 192.168.0.120:11434
   - MSSQL: 192.168.1.44:1433

## Step 1: Server Setup

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and dependencies
sudo apt install -y python3 python3-pip python3-venv
sudo apt install -y unixodbc unixodbc-dev

# Create application directory
sudo mkdir -p /opt/sisl-rag
sudo chown $USER:$USER /opt/sisl-rag
cd /opt/sisl-rag
```

## Step 2: Copy Application Files

Copy all your files to the server:

- `api.py` (your working API)
- `requirements.txt`
- All other Python files

## Step 3: Install Dependencies

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Step 4: Test the Application

```bash
# Test the API
python api.py

# In another terminal
curl http://localhost:8000/health
```

## Step 5: Production Deployment

### Option A: Simple Python (Quick Start)

```bash
# Start the API
python api.py

# Or use screen/tmux for background
screen -S rag-api
python api.py
# Press Ctrl+A, then D to detach
```

### Option B: Systemd Service (Recommended)

1. Edit the service file:

```bash
nano sisl-rag-api.service
```

2. Update paths in the service file:

```ini
[Unit]
Description=SISL RAG API
After=network.target

[Service]
Type=simple
User=your_username
WorkingDirectory=/opt/sisl-rag
Environment=PYTHONPATH=/opt/sisl-rag
ExecStart=/opt/sisl-rag/venv/bin/python api.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

3. Install and start the service:

```bash
sudo cp sisl-rag-api.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable sisl-rag-api
sudo systemctl start sisl-rag-api

# Check status
sudo systemctl status sisl-rag-api
```

### Option C: Docker (Containerized)

```bash
# Build Docker image
docker build -t sisl-rag-api .

# Run container
docker run -d \
  --name sisl-rag-api \
  -p 8000:8000 \
  --restart unless-stopped \
  sisl-rag-api
```

## Step 6: Configure Firewall

```bash
# Allow API port
sudo ufw allow 8000/tcp

# If using nginx
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
```

## Step 7: Test Deployment

```bash
# Test from server
curl http://localhost:8000/health

# Test from external machine
curl http://your-server-ip:8000/health

# Test API query
curl -X POST http://your-server-ip:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Show me top customers by sales"}'
```

## Monitoring

### Check Application Status

```bash
# If using systemd
sudo systemctl status sisl-rag-api

# Check logs
sudo journalctl -u sisl-rag-api -f

# Check if API is responding
curl http://localhost:8000/health
```

### Update Application

```bash
# Stop service
sudo systemctl stop sisl-rag-api

# Update code (copy new files)
# Then restart service
sudo systemctl start sisl-rag-api
```

## Troubleshooting

### Common Issues:

1. **Port already in use:**

```bash
sudo netstat -tulpn | grep 8000
sudo kill -9 <PID>
```

2. **Permission denied:**

```bash
sudo chown -R $USER:$USER /opt/sisl-rag
```

3. **Database connection failed:**

```bash
# Test connectivity
telnet 192.168.1.44 1433
```

4. **Qdrant/Ollama connection failed:**

```bash
# Test connectivity
telnet 192.168.0.120 6333
telnet 192.168.0.120 11434
```

### Logs Location:

- Application logs: Check terminal where API is running
- System logs: `sudo journalctl -u sisl-rag-api`

## Security Notes

1. **Change default passwords** in your database connection
2. **Use HTTPS** in production (nginx + SSL)
3. **Restrict access** to API endpoints if needed
4. **Regular backups** of your configuration

## Next Steps

1. **Set up nginx** for reverse proxy (optional)
2. **Configure SSL** certificate (recommended)
3. **Set up monitoring** (optional)
4. **Connect your Next.js frontend** to the deployed API

## Quick Commands Summary

```bash
# Start API
python api.py

# Check status (if using systemd)
sudo systemctl status sisl-rag-api

# View logs
sudo journalctl -u sisl-rag-api -f

# Test API
curl http://localhost:8000/health

# Restart service
sudo systemctl restart sisl-rag-api
```
