# 🧪 RAG API Testing Guide

## 📋 **Available APIs**

| API Version                       | Port | Status     | Description                |
| --------------------------------- | ---- | ---------- | -------------------------- |
| **Original API (Hardcoded)**      | 8000 | ✅ Working | Uses hardcoded table names |
| **Production API (DB-GPT Style)** | 8002 | 🔄 Testing | Dynamic schema discovery   |

## 🚀 **Quick Testing Commands**

### **1. Test Original API (Working)**

```bash
# Health check
curl http://localhost:8000/health

# Test query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Show me top customers by sales"}'
```

### **2. Test Production API (DB-GPT Style)**

```bash
# Health check
curl http://localhost:8002/health

# Test query
curl -X POST http://localhost:8002/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Show me top customers by sales"}'
```

### **3. PowerShell Testing**

```powershell
# Health check
Invoke-RestMethod -Uri "http://localhost:8000/health"

# Test query
Invoke-RestMethod -Uri "http://localhost:8000/query" -Method POST -ContentType "application/json" -Body '{"question": "Show me top customers by sales"}'
```

## 🧪 **Automated Testing**

### **Run Comprehensive Tests**

```bash
python test_all_apis.py
```

### **Test Specific API**

```bash
# Test original API
python test_all_apis.py "Original API (Hardcoded)"

# Test production API
python test_all_apis.py "Production API (DB-GPT Style)"
```

## 📊 **Test Questions**

### **Working Questions (Original API)**

- ✅ "Show me top customers by sales"
- ✅ "How many customers do I have?"

### **Questions That Need Fixing**

- ❌ "What are my recent orders?" (Invalid column name)
- ❌ "Count total sales this month" (Invalid column name)

## 🔍 **Manual Testing Steps**

### **Step 1: Start APIs**

```bash
# Terminal 1 - Original API
python api.py

# Terminal 2 - Production API (if needed)
python production_rag_api.py
```

### **Step 2: Test Health**

```bash
curl http://localhost:8000/health
curl http://localhost:8002/health
```

### **Step 3: Test Schema Discovery**

```bash
curl http://localhost:8000/tables
curl http://localhost:8002/schema
```

### **Step 4: Test Queries**

```bash
# Test each question
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Show me top customers by sales"}'
```

## 📈 **Expected Results**

### **Original API (Port 8000)**

```json
{
  "question": "Show me top customers by sales",
  "generated_sql": "SELECT TOP 5 c.[No_] AS [Customer Number], c.[Name] AS [Customer Name], SUM(sil.[Amount Including VAT]) AS [Total Sales] FROM [SISL Live].[dbo].[ssil_UAT$Customer] c INNER JOIN [SISL Live].[dbo].[ssil_UAT$Sales Invoice Header] sih ON c.[No_] = sih.[Sell-to Customer No_] INNER JOIN [SISL Live].[dbo].[ssil_UAT$Sales Invoice Line] sil ON sih.[No_] = sil.[Document No_] GROUP BY c.[No_], c.[Name] ORDER BY [Total Sales] DESC",
  "results": [...],
  "columns": ["Customer Number", "Customer Name", "Total Sales"],
  "row_count": 5,
  "success": true
}
```

### **Production API (Port 8002)**

```json
{
  "question": "Show me top customers by sales",
  "generated_sql": "...",
  "results": [...],
  "columns": [...],
  "row_count": 5,
  "success": true,
  "execution_time": 1.23,
  "schema_info": {
    "tables_used": ["ssil_UAT$Customer", "ssil_UAT$Sales Invoice Header"],
    "schema_context": "..."
  }
}
```

## 🐛 **Troubleshooting**

### **API Not Starting**

```bash
# Check if port is in use
netstat -ano | findstr :8000
netstat -ano | findstr :8002

# Kill process if needed
taskkill /F /PID <PID>
```

### **Connection Errors**

```bash
# Check if APIs are running
curl http://localhost:8000/health
curl http://localhost:8002/health

# Check logs
# Look for error messages in terminal where API is running
```

### **Database Connection Issues**

```bash
# Test database connection
python test_dynamic_schema.py
```

## 📝 **Testing Checklist**

- [ ] **Health Check**: Both APIs respond to `/health`
- [ ] **Schema Discovery**: Production API shows discovered tables
- [ ] **Query Success**: "Show me top customers by sales" returns 5 rows
- [ ] **SQL Generation**: Generated SQL uses correct table names
- [ ] **Error Handling**: Failed queries return proper error messages
- [ ] **Performance**: Response times under 5 seconds

## 🎯 **Next Steps**

1. **Fix Column Issues**: Update prompts to use correct column names
2. **Test Production API**: Get DB-GPT style API working
3. **Performance Testing**: Measure response times
4. **Integration Testing**: Test with Next.js frontend

## 📞 **Support**

If you encounter issues:

1. Check the logs in the terminal where the API is running
2. Verify database connection with `test_dynamic_schema.py`
3. Test individual endpoints with curl/PowerShell
4. Use the comprehensive test script: `python test_all_apis.py`
