#!/usr/bin/env python3
"""
Test script for correct RAG API deployment
"""

import requests
import json
import time

# Correct API URL (port 8000)
API_BASE_URL = "http://192.168.0.120:8000"

def test_health():
    """Test API health endpoint"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health Check: {data['status']}")
            print(f"📊 Message: {data.get('message', 'N/A')}")
            return True
        else:
            print(f"❌ Health Check Failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health Check Error: {e}")
        return False

def test_query(question):
    """Test a specific query"""
    try:
        payload = {"question": question}
        response = requests.post(
            f"{API_BASE_URL}/query",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Query: '{question}'")
            print(f"📝 SQL: {data.get('generated_sql', 'N/A')[:100]}...")
            print(f"📊 Results: {data.get('row_count', 0)} rows")
            print(f"✅ Success: {data.get('success', False)}")
            return True
        else:
            print(f"❌ Query Failed: {response.status_code}")
            print(f"📝 Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Query Error: {e}")
        return False

def main():
    """Run comprehensive tests"""
    print("🧪 TESTING CORRECT RAG API DEPLOYMENT")
    print("=" * 50)
    
    # Test health
    if not test_health():
        print("❌ API is not healthy. Deployment may have failed.")
        return
    
    print("\n🔍 Testing Queries...")
    
    # Test queries
    test_queries = [
        "How many customers do I have?",
        "Show me top customers by sales"
    ]
    
    success_count = 0
    for query in test_queries:
        print(f"\n--- Testing: {query} ---")
        if test_query(query):
            success_count += 1
        time.sleep(1)  # Small delay between requests
    
    print(f"\n📊 TEST SUMMARY")
    print(f"✅ Successful Queries: {success_count}/{len(test_queries)}")
    
    if success_count == len(test_queries):
        print("🎉 ALL TESTS PASSED! Correct API is working with proper table names.")
    else:
        print("⚠️  Some tests failed. Check the API logs for details.")

if __name__ == "__main__":
    main() 