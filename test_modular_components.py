"""
Test script for modular RAG components
Demonstrates the modular architecture working independently
"""

import asyncio
import json
from modular_rag_architecture import (
    DatabaseRAGComponent,
    DocumentRAGComponent,
    ConversationComponent,
    RAGOrchestrator,
    QueryContext,
    IntentType
)

async def test_components():
    """Test each component independently"""
    print("🧪 Testing Modular RAG Components")
    print("=" * 50)
    
    # Test Conversation Component
    print("\n1. Testing Conversation Component")
    print("-" * 30)
    conv_component = ConversationComponent()
    await conv_component.initialize()
    
    # Test relevance
    is_relevant = await conv_component.is_relevant("hi")
    print(f"Conversation relevance for 'hi': {is_relevant}")
    
    # Test processing
    context = QueryContext(
        question="hi",
        intent=IntentType.GENERAL_CONVERSATION
    )
    result = await conv_component.process_query(context)
    print(f"Conversation result: {result.content}")
    print(f"Success: {result.success}")
    
    # Test Database Component
    print("\n2. Testing Database Component")
    print("-" * 30)
    db_component = DatabaseRAGComponent()
    await db_component.initialize()
    
    # Test relevance
    is_relevant = await db_component.is_relevant("How many customers do we have?")
    print(f"Database relevance for 'How many customers do we have?': {is_relevant}")
    
    # Test processing
    context = QueryContext(
        question="How many customers do we have?",
        intent=IntentType.DATABASE_QUERY
    )
    result = await db_component.process_query(context)
    print(f"Database result success: {result.success}")
    if result.error:
        print(f"Database error: {result.error}")
    
    # Test Document Component
    print("\n3. Testing Document Component")
    print("-" * 30)
    doc_component = DocumentRAGComponent()
    await doc_component.initialize()
    
    # Test relevance
    is_relevant = await doc_component.is_relevant("What does the contract say?")
    print(f"Document relevance for 'What does the contract say?': {is_relevant}")
    
    # Test processing
    context = QueryContext(
        question="What does the contract say?",
        intent=IntentType.DOCUMENT_QUERY
    )
    result = await doc_component.process_query(context)
    print(f"Document result success: {result.success}")
    if result.error:
        print(f"Document error: {result.error}")
    
    # Test Orchestrator
    print("\n4. Testing Orchestrator")
    print("-" * 30)
    orchestrator = RAGOrchestrator()
    await orchestrator.initialize()
    
    # Test intent classification
    test_questions = [
        "hi",
        "How many customers do we have?",
        "What does the contract say?",
        "Show me customer data and related contracts"
    ]
    
    for question in test_questions:
        intent = await orchestrator.classify_intent(question)
        print(f"Question: '{question}' -> Intent: {intent.value}")
    
    # Test query processing
    context = QueryContext(
        question="hi",
        include_database=True,
        include_documents=True
    )
    results = await orchestrator.process_query(context)
    print(f"\nOrchestrator results count: {len(results)}")
    for result in results:
        print(f"  Source: {result.source}, Success: {result.success}")

async def test_api_endpoints():
    """Test API endpoints"""
    print("\n🌐 Testing API Endpoints")
    print("=" * 50)
    
    import requests
    
    base_url = "http://localhost:8001"
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/health")
        print(f"Health endpoint: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Components: {data.get('components', {})}")
    except Exception as e:
        print(f"Health endpoint error: {e}")
    
    # Test query endpoint
    try:
        response = requests.post(
            f"{base_url}/query",
            json={
                "question": "hi",
                "include_documents": True,
                "include_database": True
            }
        )
        print(f"Query endpoint: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Intent: {data.get('intent')}")
            print(f"Sources: {data.get('sources')}")
            print(f"Success: {data.get('success')}")
    except Exception as e:
        print(f"Query endpoint error: {e}")

def main():
    """Main test function"""
    print("🚀 Modular RAG Architecture Test")
    print("=" * 50)
    
    # Test components
    asyncio.run(test_components())
    
    # Test API endpoints
    asyncio.run(test_api_endpoints())
    
    print("\n✅ Testing completed!")

if __name__ == "__main__":
    main() 