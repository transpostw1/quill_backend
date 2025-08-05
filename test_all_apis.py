import requests
import json
import time
from typing import Dict, Any

# Test configurations
APIS = {
    "Original API (Hardcoded)": "http://localhost:8000",
    "Production API (DB-GPT Style)": "http://localhost:8002"
}

# Test questions
TEST_QUESTIONS = [
    "Show me top customers by sales",
    "How many customers do I have?",
    "What are my recent orders?",
    "Count total sales this month"
]

def test_api_health(api_name: str, base_url: str) -> bool:
    """Test if API is healthy"""
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print(f"✅ {api_name}: Healthy")
            return True
        else:
            print(f"❌ {api_name}: Unhealthy (Status: {response.status_code})")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ {api_name}: Connection failed - {e}")
        return False

def test_api_query(api_name: str, base_url: str, question: str) -> Dict[str, Any]:
    """Test a specific query on an API"""
    try:
        payload = {"question": question}
        response = requests.post(
            f"{base_url}/query",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return {
                "success": result.get("success", False),
                "sql": result.get("generated_sql", ""),
                "row_count": result.get("row_count", 0),
                "error": result.get("error"),
                "execution_time": result.get("execution_time")
            }
        else:
            return {
                "success": False,
                "sql": "",
                "row_count": 0,
                "error": f"HTTP {response.status_code}",
                "execution_time": None
            }
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "sql": "",
            "row_count": 0,
            "error": f"Connection error: {e}",
            "execution_time": None
        }

def test_api_schema(api_name: str, base_url: str) -> Dict[str, Any]:
    """Test schema endpoint"""
    try:
        response = requests.get(f"{base_url}/schema", timeout=10)
        if response.status_code == 200:
            schema = response.json()
            return {
                "success": True,
                "tables": schema.get("total_tables", 0),
                "relationships": schema.get("total_relationships", 0)
            }
        else:
            return {
                "success": False,
                "tables": 0,
                "relationships": 0
            }
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "tables": 0,
            "relationships": 0
        }

def run_comprehensive_test():
    """Run comprehensive tests on all APIs"""
    print("🚀 COMPREHENSIVE RAG API TESTING")
    print("=" * 50)
    
    results = {}
    
    for api_name, base_url in APIS.items():
        print(f"\n🔍 Testing: {api_name}")
        print("-" * 30)
        
        # Test health
        is_healthy = test_api_health(api_name, base_url)
        
        if not is_healthy:
            print(f"⏭️  Skipping {api_name} - not healthy")
            results[api_name] = {"status": "unhealthy"}
            continue
        
        # Test schema discovery
        schema_result = test_api_schema(api_name, base_url)
        if schema_result["success"]:
            print(f"📊 Schema: {schema_result['tables']} tables, {schema_result['relationships']} relationships")
        else:
            print("📊 Schema: Not available")
        
        # Test queries
        query_results = {}
        for question in TEST_QUESTIONS:
            print(f"\n🤖 Testing: '{question}'")
            result = test_api_query(api_name, base_url, question)
            
            if result["success"]:
                print(f"   ✅ Success: {result['row_count']} rows")
                print(f"   📝 SQL: {result['sql'][:100]}...")
                if result.get("execution_time"):
                    print(f"   ⏱️  Time: {result['execution_time']:.2f}s")
            else:
                print(f"   ❌ Failed: {result['error']}")
            
            query_results[question] = result
        
        results[api_name] = {
            "status": "healthy",
            "schema": schema_result,
            "queries": query_results
        }
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    for api_name, result in results.items():
        print(f"\n🔍 {api_name}")
        if result["status"] == "healthy":
            successful_queries = sum(1 for q in result["queries"].values() if q["success"])
            total_queries = len(result["queries"])
            print(f"   ✅ Status: Healthy")
            print(f"   📊 Schema: {result['schema']['tables']} tables")
            print(f"   🎯 Queries: {successful_queries}/{total_queries} successful")
            
            # Show best performing query
            best_query = max(result["queries"].values(), key=lambda x: x.get("row_count", 0))
            if best_query["success"]:
                print(f"   🏆 Best result: {best_query['row_count']} rows")
        else:
            print(f"   ❌ Status: Unhealthy")

def test_specific_api(api_name: str, base_url: str):
    """Test a specific API in detail"""
    print(f"\n🎯 DETAILED TEST: {api_name}")
    print("=" * 40)
    
    # Health check
    if not test_api_health(api_name, base_url):
        return
    
    # Test each question
    for question in TEST_QUESTIONS:
        print(f"\n🤖 Question: {question}")
        result = test_api_query(api_name, base_url, question)
        
        if result["success"]:
            print(f"   ✅ Success!")
            print(f"   📊 Rows: {result['row_count']}")
            print(f"   📝 SQL: {result['sql']}")
            if result.get("execution_time"):
                print(f"   ⏱️  Time: {result['execution_time']:.2f}s")
        else:
            print(f"   ❌ Failed: {result['error']}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Test specific API
        api_name = sys.argv[1]
        if api_name in APIS:
            test_specific_api(api_name, APIS[api_name])
        else:
            print(f"❌ Unknown API: {api_name}")
            print(f"Available APIs: {list(APIS.keys())}")
    else:
        # Run comprehensive test
        run_comprehensive_test() 