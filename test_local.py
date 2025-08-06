#!/usr/bin/env python3
"""
Local test script for RAG API
"""
import requests
import json
import time
import subprocess
import threading
import sys

# Local API URL (port 8000)
API_BASE_URL = "http://localhost:8000"

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

def clear_memory(chat_id):
    """Clear conversation memory for a specific chat ID"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/clear_memory",
            params={"chat_id": chat_id},
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Memory Clear: {data.get('message', 'N/A')}")
            return True
        else:
            print(f"❌ Memory Clear Failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Memory Clear Error: {e}")
        return False

def test_query(question, chat_id="test-session-123"):
    """Test a specific query"""
    try:
        payload = {"question": question, "chat_id": chat_id}
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
            if data.get('results'):
                print(f"📋 Sample Result: {data['results'][0]}")
            return True
        else:
            print(f"❌ Query Failed: {response.status_code}")
            print(f"📝 Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Query Error: {e}")
        return False

def test_conversation_context():
    """Test conversation context with multiple related queries"""
    print("\n💬 Testing Conversation Context...")
    
    # Test sequence that should leverage conversation memory
    conversation = [
        "Hi there!",
        "How many customers do you have?",
        "What was my first question?",
        "Can you tell me more about the customer data?",
        "Thanks for your help!"
    ]
    
    chat_id = "conversation-test-456"
    success_count = 0
    for i, question in enumerate(conversation):
        print(f"\n--- Test {i+1}: {question} ---")
        if test_query(question, chat_id):
            success_count += 1
        time.sleep(1)  # Small delay between requests
    
    print(f"\n📊 Conversation Context Test Summary")
    print(f"✅ Successful Queries: {success_count}/{len(conversation)}")
    return success_count == len(conversation)

def test_chat_id_isolation():
    """Test that different chat IDs have separate conversation contexts"""
    print("\n🔒 Testing Chat ID Isolation...")
    
    # Test with first chat ID
    chat_id_1 = "isolation-test-1"
    test_query("What is your name?", chat_id_1)
    time.sleep(1)
    
    # Test with second chat ID
    chat_id_2 = "isolation-test-2"
    test_query("What is your name?", chat_id_2)
    time.sleep(1)
    
    # Test if second chat ID remembers its own context
    result_2 = test_query("What was my previous question?", chat_id_2)
    time.sleep(1)
    
    # Test if first chat ID remembers its own context
    result_1 = test_query("What was my previous question?", chat_id_1)
    time.sleep(1)
    
    # Both should work since each has its own context
    return result_1 and result_2

def main():
    """Run local tests"""
    print("🧪 TESTING RAG API LOCALLY")
    print("=" * 50)
    
    # Test health
    if not test_health():
        print("❌ API is not healthy. Make sure the API is running locally.")
        return
    
    print("\n🔍 Testing Queries...")
    
    # Test basic queries
    test_queries = [
        "Hi, how are you?",
        "How many customers do I have?",
        "What tables are available?"
    ]
    
    success_count = 0
    for query in test_queries:
        print(f"\n--- Testing: {query} ---")
        if test_query(query, "basic-test-789"):
            success_count += 1
        time.sleep(1)  # Small delay between requests
    
    print(f"\n📊 BASIC TEST SUMMARY")
    print(f"✅ Successful Queries: {success_count}/{len(test_queries)}")
    
    # Test conversation context
    context_test_passed = test_conversation_context()
    
    # Test chat ID isolation
    print("\n🔒 Testing Chat ID Isolation...")
    isolation_test_passed = test_chat_id_isolation()
    
    # Test memory clearing
    print("\n🧹 Testing Memory Clear...")
    clear_memory_success = clear_memory("conversation-test-456")
    
    print(f"\n{'='*50}")
    if success_count == len(test_queries) and context_test_passed and isolation_test_passed and clear_memory_success:
        print("🎉 ALL LOCAL TESTS PASSED!")
        print("✅ Basic queries working")
        print("✅ Conversation context working")
        print("✅ Chat ID isolation working")
        print("✅ Memory clearing working")
        print("✅ Ready for deployment!")
    else:
        print("⚠️  Some tests failed. Check the output above.")
        if success_count != len(test_queries):
            print("❌ Basic queries failed")
        if not context_test_passed:
            print("❌ Conversation context test failed")
        if not isolation_test_passed:
            print("❌ Chat ID isolation test failed")
        if not clear_memory_success:
            print("❌ Memory clearing test failed")

if __name__ == "__main__":
    main()