#!/usr/bin/env python3
"""
Test script for Document RAG System
This script tests the main functionality of the document RAG API
"""

import requests
import json
import time
import tempfile
import os
from pathlib import Path

# Configuration
API_BASE_URL = "http://localhost:8000"
TEST_DOCUMENT_CONTENT = """
Artificial Intelligence and Machine Learning

Artificial Intelligence (AI) is a broad field of computer science that aims to create systems capable of performing tasks that typically require human intelligence. These tasks include learning, reasoning, problem-solving, perception, and language understanding.

Machine Learning is a subset of AI that focuses on the development of algorithms and statistical models that enable computers to improve their performance on a specific task through experience. Instead of being explicitly programmed for every scenario, machine learning systems learn patterns from data.

Key Concepts in Machine Learning:

1. Supervised Learning: The algorithm learns from labeled training data to make predictions on new, unseen data. Examples include classification and regression tasks.

2. Unsupervised Learning: The algorithm finds hidden patterns in data without any labels. Examples include clustering and dimensionality reduction.

3. Deep Learning: A subset of machine learning that uses neural networks with multiple layers to model and understand complex patterns in data.

4. Natural Language Processing (NLP): A field that focuses on the interaction between computers and human language, enabling machines to understand, interpret, and generate human language.

Applications of AI and ML:
- Computer Vision: Image and video recognition
- Speech Recognition: Converting speech to text
- Recommendation Systems: Suggesting products or content
- Autonomous Vehicles: Self-driving cars
- Healthcare: Disease diagnosis and drug discovery
- Finance: Fraud detection and algorithmic trading

The future of AI and ML holds tremendous potential for transforming industries and improving human lives through automation, personalization, and intelligent decision-making systems.
"""

def create_test_document():
    """Create a test document file"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(TEST_DOCUMENT_CONTENT)
        return f.name

def test_health_check():
    """Test the health check endpoint"""
    print("🔍 Testing health check...")
    try:
        response = requests.get(f"{API_BASE_URL}/")
        if response.status_code == 200:
            health = response.json()
            print(f"✅ Health check passed: {health.get('status')}")
            print(f"   Components: {health.get('components')}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_document_upload():
    """Test document upload functionality"""
    print("\n📤 Testing document upload...")
    
    # Create test document
    test_file = create_test_document()
    
    try:
        with open(test_file, 'rb') as f:
            files = {'file': ('test_ai_document.txt', f, 'text/plain')}
            data = {
                'description': 'Test document about AI and Machine Learning',
                'tags': json.dumps(['ai', 'machine-learning', 'test']),
                'metadata': json.dumps({'category': 'technology', 'test': True})
            }
            
            response = requests.post(f"{API_BASE_URL}/upload", files=files, data=data)
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Document uploaded successfully!")
                print(f"   Document ID: {result.get('document_id')}")
                print(f"   Status: {result.get('status')}")
                return result.get('document_id')
            else:
                print(f"❌ Upload failed: {response.status_code} - {response.text}")
                return None
                
    except Exception as e:
        print(f"❌ Upload error: {e}")
        return None
    finally:
        # Clean up test file
        if os.path.exists(test_file):
            os.unlink(test_file)

def test_document_search(document_id=None):
    """Test document search functionality"""
    print("\n🔍 Testing document search...")
    
    # Wait a bit for processing
    print("   Waiting for document processing...")
    time.sleep(5)
    
    search_queries = [
        "What is artificial intelligence?",
        "Explain machine learning",
        "What are the applications of AI?",
        "How does deep learning work?"
    ]
    
    for query in search_queries:
        try:
            payload = {
                "query": query,
                "max_results": 3,
                "similarity_threshold": 0.7,
                "include_metadata": True
            }
            
            response = requests.post(f"{API_BASE_URL}/search", json=payload)
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Search for '{query}': {result.get('total_results')} results")
                if result.get('results'):
                    first_result = result['results'][0]
                    print(f"   Top result score: {first_result.get('score', 'N/A')}")
            else:
                print(f"❌ Search failed for '{query}': {response.status_code}")
                
        except Exception as e:
            print(f"❌ Search error for '{query}': {e}")

def test_chat_functionality():
    """Test chat functionality"""
    print("\n💬 Testing chat functionality...")
    
    chat_questions = [
        "What is the difference between AI and machine learning?",
        "What are the main types of machine learning?",
        "How is AI used in healthcare?",
        "What is natural language processing?"
    ]
    
    chat_history = []
    
    for question in chat_questions:
        try:
            payload = {
                "question": question,
                "chat_history": chat_history,
                "max_context_length": 4000
            }
            
            response = requests.post(f"{API_BASE_URL}/chat", json=payload)
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get('answer', 'No answer received')
                sources = result.get('sources', [])
                processing_time = result.get('processing_time', 0)
                
                print(f"✅ Q: {question}")
                print(f"   A: {answer[:100]}...")
                print(f"   Sources: {len(sources)} documents")
                print(f"   Time: {processing_time:.2f}s")
                
                # Update chat history
                chat_history.append({"role": "user", "content": question})
                chat_history.append({"role": "assistant", "content": answer})
                
            else:
                print(f"❌ Chat failed for '{question}': {response.status_code}")
                
        except Exception as e:
            print(f"❌ Chat error for '{question}': {e}")

def test_document_management():
    """Test document management functionality"""
    print("\n📋 Testing document management...")
    
    try:
        # List documents
        response = requests.get(f"{API_BASE_URL}/documents")
        
        if response.status_code == 200:
            documents = response.json()
            print(f"✅ Found {len(documents)} documents")
            for doc in documents:
                print(f"   - {doc.get('filename', 'Unknown')} (ID: {doc.get('document_id', 'N/A')})")
        else:
            print(f"❌ Failed to list documents: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Document management error: {e}")

def main():
    """Run all tests"""
    print("🧪 Document RAG System Test Suite")
    print("=" * 50)
    
    # Test health check
    if not test_health_check():
        print("❌ Health check failed. Make sure the API is running.")
        return
    
    # Test document upload
    document_id = test_document_upload()
    
    # Test search functionality
    test_document_search(document_id)
    
    # Test chat functionality
    test_chat_functionality()
    
    # Test document management
    test_document_management()
    
    print("\n" + "=" * 50)
    print("✅ Test suite completed!")
    print("\n📖 Next steps:")
    print("1. Open document_rag_interface.html in your browser")
    print("2. Try uploading your own documents")
    print("3. Explore the search and chat features")
    print("4. Check the API documentation at http://localhost:8000/docs")

if __name__ == "__main__":
    main() 