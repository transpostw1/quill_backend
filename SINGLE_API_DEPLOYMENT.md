# 🚀 Single Correct RAG API Deployment

## 🎯 **Goal: One Correct API**

We're consolidating to use only the **Correct RAG API** (`api.py`) which has:

- ✅ Correct table names (`ssil_UAT$Customer`, etc.)
- ✅ Working queries with proper SQL generation
- ✅ Stable and tested functionality
- ✅ Simple and reliable architecture

## 📋 **Deployment Steps**

### **1. Upload Files to Server**

```bash
# From your local machine
scp api.py root@192.168.0.120:/home/root/rag-api/
scp sisl-rag-api.service root@192.168.0.120:/home/root/rag-api/
scp redeploy_production_api.sh root@192.168.0.120:/home/root/rag-api/
```

### **2. Run Redeployment Script**

```bash
# On your Ubuntu server
cd /home/root/rag-api
chmod +x redeploy_production_api.sh
./redeploy_production_api.sh
```

### **3. Verify Deployment**

```bash
# Check service status
sudo systemctl status sisl-rag-api

# View logs
sudo journalctl -u sisl-rag-api -f

# Test the API
curl http://192.168.0.120:8000/health
```

## 🧪 **Testing the Deployment**

### **Quick Test**

```bash
curl -X POST http://192.168.0.120:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How many customers do I have?"}'
```

### **Comprehensive Test**

```bash
# Run the test script
python test_production_deployment.py
```

## 📊 **API Endpoints**

| Endpoint  | Method | Description             |
| --------- | ------ | ----------------------- |
| `/health` | GET    | Health check            |
| `/query`  | POST   | Main RAG query endpoint |
| `/tables` | GET    | Get available tables    |

## 🔧 **Service Management**

```bash
# Check status
sudo systemctl status sisl-rag-api

# Restart service
sudo systemctl restart sisl-rag-api

# View logs
sudo journalctl -u sisl-rag-api -f

# Stop service
sudo systemctl stop sisl-rag-api
```

## 🎯 **Expected Results**

### **Health Check Response**

```json
{
  "status": "healthy",
  "message": "SISL RAG API is running"
}
```

### **Query Response**

```json
{
  "question": "How many customers do I have?",
  "generated_sql": "SELECT COUNT(*) FROM [SISL Live].[dbo].[ssil_UAT$Customer]",
  "results": [{ "": "123" }],
  "columns": [""],
  "row_count": 1,
  "success": true,
  "error": null
}
```

## 🚨 **Troubleshooting**

### **If Service Won't Start**

```bash
# Check logs
sudo journalctl -u sisl-rag-api -n 50

# Test manually
cd /home/root/rag-api
source rag_env/bin/activate
python api.py
```

### **If API Returns 422 Errors**

- Check JSON payload format
- Ensure all required fields are present
- Verify Content-Type header

### **If Database Connection Fails**

```bash
# Test database connection
python test.py
```

## ✅ **Success Criteria**

- [ ] Service starts without errors
- [ ] Health endpoint returns 200
- [ ] Query endpoint processes requests
- [ ] SQL generation uses correct table names
- [ ] Database queries execute successfully
- [ ] No more 422 errors in logs

## 🎉 **Benefits of Correct API**

1. **Correct Table Names**: Uses `ssil_UAT$Customer` instead of wrong names
2. **Working Queries**: Successfully executes customer count queries
3. **Stable Performance**: Proven to work in your environment
4. **Simple Maintenance**: Straightforward codebase
5. **Reliable Results**: Consistent behavior

---

**🌐 API URL**: `http://192.168.0.120:8000`
**📧 Support**: Check logs with `sudo journalctl -u sisl-rag-api -f`
