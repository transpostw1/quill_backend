# Document RAG System Deployment Script for Windows
# This script sets up and deploys the Document RAG API

param(
    [string]$QdrantUrl = "http://192.168.0.120:6333",
    [string]$OllamaUrl = "http://192.168.0.120:11434",
    [int]$ApiPort = 8000
)

# Configuration
$CollectionName = "document_vectors"
$EmbeddingModel = "nomic-embed-text"
$LlmModel = "llama3.2:3b"

# Function to write colored output
function Write-Status {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Blue
}

function Write-Success {
    param([string]$Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

# Check if running as administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")
if ($isAdmin) {
    Write-Warning "This script should not be run as administrator"
    exit 1
}

Write-Host "🚀 Deploying Document RAG System..." -ForegroundColor Cyan

# Check Python version
Write-Status "Checking Python version..."
try {
    $pythonVersion = python --version 2>&1
    if ($pythonVersion -match "Python (\d+\.\d+)") {
        $version = $matches[1]
        if ([version]$version -ge [version]"3.8") {
            Write-Success "Python $version is compatible"
        }
        else {
            Write-Error "Python $version is not compatible. Required: 3.8 or higher"
            exit 1
        }
    }
    else {
        Write-Error "Could not determine Python version"
        exit 1
    }
}
catch {
    Write-Error "Python is not installed or not in PATH"
    exit 1
}

# Check if virtual environment exists
if (-not (Test-Path "rag_env")) {
    Write-Status "Creating virtual environment..."
    python -m venv rag_env
    Write-Success "Virtual environment created"
}

# Activate virtual environment
Write-Status "Activating virtual environment..."
& "rag_env\Scripts\Activate.ps1"

# Upgrade pip
Write-Status "Upgrading pip..."
python -m pip install --upgrade pip

# Install dependencies
Write-Status "Installing dependencies..."
pip install -r requirements.txt

# Install additional dependencies for document processing
Write-Status "Installing document processing dependencies..."
pip install unstructured python-magic-bin

# Check if Qdrant is accessible
Write-Status "Checking Qdrant connection..."
try {
    $response = Invoke-RestMethod -Uri "$QdrantUrl/collections" -Method Get -TimeoutSec 5
    Write-Success "Qdrant is accessible"
}
catch {
    Write-Warning "Qdrant is not accessible at $QdrantUrl"
    Write-Warning "Make sure Qdrant is running and accessible"
}

# Check if Ollama is accessible
Write-Status "Checking Ollama connection..."
try {
    $response = Invoke-RestMethod -Uri "$OllamaUrl/api/tags" -Method Get -TimeoutSec 5
    Write-Success "Ollama is accessible"
}
catch {
    Write-Warning "Ollama is not accessible at $OllamaUrl"
    Write-Warning "Make sure Ollama is running and accessible"
}

# Create necessary directories
Write-Status "Creating directories..."
New-Item -ItemType Directory -Force -Path "temp_documents" | Out-Null
New-Item -ItemType Directory -Force -Path "logs" | Out-Null

# Check if the collection exists, if not create it
Write-Status "Checking Qdrant collection..."
try {
    $response = Invoke-RestMethod -Uri "$QdrantUrl/collections/$CollectionName" -Method Get -TimeoutSec 5
    Write-Success "Collection already exists"
}
catch {
    Write-Status "Creating Qdrant collection: $CollectionName"
    $collectionConfig = @{
        vectors = @{
            size     = 768
            distance = "Cosine"
        }
    } | ConvertTo-Json -Depth 3
    
    try {
        Invoke-RestMethod -Uri "$QdrantUrl/collections/$CollectionName" -Method Put -Body $collectionConfig -ContentType "application/json"
        Write-Success "Collection created"
    }
    catch {
        Write-Warning "Failed to create collection: $($_.Exception.Message)"
    }
}

# Create Windows service (optional - requires admin privileges)
Write-Status "Creating Windows service configuration..."
$serviceConfig = @"
# Windows Service Configuration for Document RAG
# To install as a Windows service, run the following commands as Administrator:

# 1. Install NSSM (if not already installed):
#    choco install nssm

# 2. Create the service:
#    nssm install DocumentRAG "$(Get-Location)\rag_env\Scripts\python.exe" "$(Get-Location)\document_rag_api.py"
#    nssm set DocumentRAG AppDirectory "$(Get-Location)"
#    nssm set DocumentRAG Description "Document RAG API Service"
#    nssm set DocumentRAG Start SERVICE_AUTO_START

# 3. Start the service:
#    nssm start DocumentRAG

# 4. Check service status:
#    nssm status DocumentRAG
"@

$serviceConfig | Out-File -FilePath "document_rag_service_config.txt" -Encoding UTF8
Write-Success "Service configuration saved to document_rag_service_config.txt"

# Start the API manually
Write-Status "Starting Document RAG API..."
$apiProcess = Start-Process -FilePath "rag_env\Scripts\python.exe" -ArgumentList "document_rag_api.py" -PassThru -WindowStyle Hidden

# Wait a moment for the service to start
Start-Sleep -Seconds 5

# Check if the process is running
if (-not $apiProcess.HasExited) {
    Write-Success "Document RAG API is running (PID: $($apiProcess.Id))"
}
else {
    Write-Error "Failed to start Document RAG API"
    exit 1
}

# Check API health
Write-Status "Checking API health..."
Start-Sleep -Seconds 2
try {
    $healthResponse = Invoke-RestMethod -Uri "http://localhost:$ApiPort/" -Method Get -TimeoutSec 5
    Write-Success "API is responding"
    Write-Host "   Status: $($healthResponse.status)"
    Write-Host "   Message: $($healthResponse.message)"
}
catch {
    Write-Warning "API is not responding yet, it may still be starting up"
}

# Display service information
Write-Success "Deployment completed!"
Write-Host ""
Write-Host "📋 Service Information:" -ForegroundColor Cyan
Write-Host "  API URL: http://localhost:$ApiPort"
Write-Host "  API Docs: http://localhost:$ApiPort/docs"
Write-Host "  Health Check: http://localhost:$ApiPort/"
Write-Host ""
Write-Host "🔧 Management Commands:" -ForegroundColor Cyan
Write-Host "  Start API:   Start-Process -FilePath 'rag_env\Scripts\python.exe' -ArgumentList 'document_rag_api.py'"
Write-Host "  Stop API:    Stop-Process -Name 'python' -Force"
Write-Host "  Check Status: Get-Process -Name 'python'"
Write-Host ""
Write-Host "🌐 Web Interface:" -ForegroundColor Cyan
Write-Host "  Open document_rag_interface.html in your browser"
Write-Host ""
Write-Host "📁 Files:" -ForegroundColor Cyan
Write-Host "  API: document_rag_api.py"
Write-Host "  Client: document_rag_client.py"
Write-Host "  Interface: document_rag_interface.html"
Write-Host "  Test: test_document_rag.py"
Write-Host ""

# Test the API
Write-Status "Running quick API test..."
try {
    python test_document_rag.py
}
catch {
    Write-Warning "Test failed: $($_.Exception.Message)"
}

Write-Success "Document RAG System is ready to use!"
Write-Host ""
Write-Host "💡 Quick Start:" -ForegroundColor Yellow
Write-Host "1. Open document_rag_interface.html in your browser"
Write-Host "2. Upload a document using the web interface"
Write-Host "3. Try searching and chatting with your documents"
Write-Host "4. Check the API documentation at http://localhost:$ApiPort/docs" 