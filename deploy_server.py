#!/usr/bin/env python3
"""
Production Deployment Script for SISL RAG API
Deploy this on your server for production use
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Production configuration
PRODUCTION_CONFIG = {
    "host": "0.0.0.0",  # Listen on all interfaces
    "port": 8000,       # Production port
    "workers": 4,        # Number of worker processes
    "reload": False,     # Disable auto-reload in production
    "log_level": "info"
}

def create_production_script():
    """Create a production startup script"""
    
    script_content = '''#!/bin/bash
# Production startup script for SISL RAG API

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export RAG_ENV="production"

# Activate virtual environment (if using one)
# source /path/to/your/venv/bin/activate

# Start the API with production settings
python api.py
'''
    
    with open("start_production.sh", "w") as f:
        f.write(script_content)
    
    # Make executable
    os.chmod("start_production.sh", 0o755)
    logger.info("✅ Created production startup script: start_production.sh")

def create_systemd_service():
    """Create systemd service file for auto-start"""
    
    service_content = '''[Unit]
Description=SISL RAG API
After=network.target

[Service]
Type=simple
User=your_username
WorkingDirectory=/path/to/your/rag/api
Environment=PYTHONPATH=/path/to/your/rag/api
ExecStart=/path/to/your/venv/bin/python api.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
'''
    
    with open("sisl-rag-api.service", "w") as f:
        f.write(service_content)
    
    logger.info("✅ Created systemd service file: sisl-rag-api.service")
    logger.info("📝 Edit the paths in the service file and run:")
    logger.info("   sudo cp sisl-rag-api.service /etc/systemd/system/")
    logger.info("   sudo systemctl enable sisl-rag-api")
    logger.info("   sudo systemctl start sisl-rag-api")

def create_dockerfile():
    """Create Dockerfile for containerized deployment"""
    
    dockerfile_content = '''FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    unixodbc \\
    unixodbc-dev \\
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Start the application
CMD ["python", "api.py"]
'''
    
    with open("Dockerfile", "w") as f:
        f.write(dockerfile_content)
    
    logger.info("✅ Created Dockerfile for containerized deployment")

def create_requirements():
    """Create requirements.txt for production dependencies"""
    
    requirements = '''fastapi==0.104.1
uvicorn[standard]==0.24.0
langchain==0.1.0
langchain-community==0.0.10
langchain-ollama==0.1.0
langchain-qdrant==0.1.0
pyodbc==5.0.1
qdrant-client==1.7.0
pydantic==2.5.0
requests==2.31.0
python-multipart==0.0.6
'''
    
    with open("requirements.txt", "w") as f:
        f.write(requirements)
    
    logger.info("✅ Created requirements.txt")

def create_nginx_config():
    """Create nginx configuration for reverse proxy"""
    
    nginx_config = '''server {
    listen 80;
    server_name your-domain.com;  # Replace with your domain

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support (if needed)
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
'''
    
    with open("nginx-sisl-rag.conf", "w") as f:
        f.write(nginx_config)
    
    logger.info("✅ Created nginx configuration: nginx-sisl-rag.conf")
    logger.info("📝 Copy to /etc/nginx/sites-available/ and enable")

def create_environment_file():
    """Create environment configuration template"""
    
    env_template = '''# Production Environment Variables
RAG_ENV=production

# Database Configuration
DB_SERVER=192.168.1.44,1433
DB_NAME=SISL Live
DB_USER=test
DB_PASSWORD=Test@345

# External Services
QDRANT_URL=http://192.168.0.120:6333/
OLLAMA_URL=http://192.168.0.120:11434

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Security (for production)
CORS_ORIGINS=["https://your-domain.com"]
'''
    
    with open(".env.production", "w") as f:
        f.write(env_template)
    
    logger.info("✅ Created environment template: .env.production")

def create_deployment_guide():
    """Create comprehensive deployment guide"""
    
    guide = '''# 🚀 Production Deployment Guide

## 📋 Prerequisites

1. **Server Requirements:**
   - Ubuntu 20.04+ or CentOS 8+
   - Python 3.9+
   - 4GB+ RAM
   - 20GB+ disk space

2. **External Services:**
   - Qdrant running on 192.168.0.120:6333
   - Ollama running on 192.168.0.120:11434
   - MSSQL database accessible

## 🔧 Installation Steps

### Step 1: Server Setup
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

### Step 2: Application Setup
```bash
# Clone or copy your application
# Copy all files to /opt/sisl-rag/

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Configuration
```bash
# Copy and edit environment file
cp .env.production .env
nano .env  # Edit with your actual values
```

### Step 4: Test Application
```bash
# Test the API
python api.py

# In another terminal
curl http://localhost:8000/health
```

## 🚀 Deployment Options

### Option A: Direct Python (Simple)
```bash
# Start the API
python api.py

# Or use the production script
./start_production.sh
```

### Option B: Systemd Service (Recommended)
```bash
# Edit the service file with correct paths
nano sisl-rag-api.service

# Install the service
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
docker run -d \\
  --name sisl-rag-api \\
  -p 8000:8000 \\
  --restart unless-stopped \\
  sisl-rag-api
```

### Option D: Nginx Reverse Proxy
```bash
# Install nginx
sudo apt install nginx

# Copy nginx configuration
sudo cp nginx-sisl-rag.conf /etc/nginx/sites-available/sisl-rag
sudo ln -s /etc/nginx/sites-available/sisl-rag /etc/nginx/sites-enabled/

# Test and reload nginx
sudo nginx -t
sudo systemctl reload nginx
```

## 🔍 Monitoring & Maintenance

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

# Update code
git pull  # or copy new files

# Restart service
sudo systemctl start sisl-rag-api
```

### Backup Configuration
```bash
# Backup important files
sudo cp /opt/sisl-rag/api.py /backup/
sudo cp /opt/sisl-rag/.env /backup/
```

## 🔒 Security Considerations

1. **Firewall Configuration:**
   ```bash
   sudo ufw allow 8000/tcp  # API port
   sudo ufw allow 80/tcp     # HTTP (if using nginx)
   sudo ufw allow 443/tcp    # HTTPS (if using SSL)
   ```

2. **SSL Certificate (Recommended):**
   ```bash
   # Install certbot
   sudo apt install certbot python3-certbot-nginx
   
   # Get SSL certificate
   sudo certbot --nginx -d your-domain.com
   ```

3. **Environment Security:**
   - Use strong passwords in .env file
   - Restrict file permissions: `chmod 600 .env`
   - Use environment variables for secrets

## 📊 Performance Tuning

### For High Traffic:
```bash
# Increase workers in api.py
uvicorn.run(app, host="0.0.0.0", port=8000, workers=8)

# Or use gunicorn
pip install gunicorn
gunicorn -w 8 -k uvicorn.workers.UvicornWorker api:app
```

### Database Connection Pooling:
```python
# In api.py, add connection pooling
import pyodbc
from contextlib import contextmanager

@contextmanager
def get_db_connection():
    conn = pyodbc.connect(MSSQL_CONN_STR, autocommit=True)
    try:
        yield conn
    finally:
        conn.close()
```

## 🆘 Troubleshooting

### Common Issues:
1. **Port already in use:** `sudo netstat -tulpn | grep 8000`
2. **Permission denied:** `sudo chown -R $USER:$USER /opt/sisl-rag`
3. **Database connection failed:** Check network connectivity to 192.168.1.44
4. **Qdrant/Ollama connection failed:** Check services on 192.168.0.120

### Logs Location:
- Application logs: `/var/log/sisl-rag-api.log`
- System logs: `sudo journalctl -u sisl-rag-api`
- Nginx logs: `/var/log/nginx/access.log`

## 📞 Support

For issues:
1. Check application logs
2. Verify external service connectivity
3. Test individual components
4. Review this deployment guide
'''
    
    with open("DEPLOYMENT_GUIDE.md", "w") as f:
        f.write(guide)
    
    logger.info("✅ Created comprehensive deployment guide: DEPLOYMENT_GUIDE.md")

def main():
    """Create all deployment files"""
    logger.info("🚀 Creating production deployment files...")
    
    create_production_script()
    create_systemd_service()
    create_dockerfile()
    create_requirements()
    create_nginx_config()
    create_environment_file()
    create_deployment_guide()
    
    logger.info("\n🎉 Deployment files created successfully!")
    logger.info("\n📋 Next steps:")
    logger.info("1. Copy all files to your server")
    logger.info("2. Follow DEPLOYMENT_GUIDE.md")
    logger.info("3. Choose your deployment method")
    logger.info("4. Test the deployment")

if __name__ == "__main__":
    main() 