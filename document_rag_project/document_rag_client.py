import requests
import json
import time
from pathlib import Path
from typing import Dict, Any, List

class DocumentRAGClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def health_check(self) -> Dict[str, Any]:
        """Check API health"""
        try:
            response = self.session.get(f"{self.base_url}/")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def upload_document(self, file_path: str, description: str = None, tags: List[str] = None, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Upload a document for processing"""
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Prepare form data
            files = {'file': (file_path.name, open(file_path, 'rb'))}
            data = {}
            
            if description:
                data['description'] = description
            if tags:
                data['tags'] = json.dumps(tags)
            if metadata:
                data['metadata'] = json.dumps(metadata)
            
            response = self.session.post(f"{self.base_url}/upload", files=files, data=data)
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            return {"error": str(e)}
    
    def search_documents(self, query: str, max_results: int = 10, similarity_threshold: float = 0.7, filters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Search documents using semantic similarity"""
        try:
            payload = {
                "query": query,
                "max_results": max_results,
                "similarity_threshold": similarity_threshold,
                "include_metadata": True
            }
            
            if filters:
                payload["filters"] = filters
            
            response = self.session.post(f"{self.base_url}/search", json=payload)
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            return {"error": str(e)}
    
    def chat_with_documents(self, question: str, chat_history: List[Dict[str, str]] = None, max_context_length: int = 4000) -> Dict[str, Any]:
        """Chat with documents using RAG"""
        try:
            payload = {
                "question": question,
                "max_context_length": max_context_length
            }
            
            if chat_history:
                payload["chat_history"] = chat_history
            
            response = self.session.post(f"{self.base_url}/chat", json=payload)
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            return {"error": str(e)}
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """List all uploaded documents"""
        try:
            response = self.session.get(f"{self.base_url}/documents")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return [{"error": str(e)}]
    
    def delete_document(self, document_id: str) -> Dict[str, Any]:
        """Delete a document"""
        try:
            response = self.session.delete(f"{self.base_url}/documents/{document_id}")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}

def main():
    """Example usage of the Document RAG Client"""
    client = DocumentRAGClient()
    
    print("=== Document RAG Client Example ===\n")
    
    # 1. Health check
    print("1. Checking API health...")
    health = client.health_check()
    print(f"Health status: {health.get('status', 'unknown')}")
    print(f"Message: {health.get('message', 'N/A')}")
    print(f"Components: {health.get('components', {})}")
    print()
    
    # 2. Upload a document (example with a text file)
    print("2. Uploading a sample document...")
    
    # Create a sample document for testing
    sample_content = """
    Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines.
    These machines can perform tasks that typically require human intelligence, such as visual perception,
    speech recognition, decision-making, and language translation.
    
    Machine Learning is a subset of AI that enables computers to learn and improve from experience
    without being explicitly programmed. It uses algorithms to identify patterns in data and make
    predictions or decisions based on those patterns.
    
    Deep Learning is a subset of machine learning that uses neural networks with multiple layers
    to model and understand complex patterns in data. It has been particularly successful in
    image recognition, natural language processing, and speech recognition.
    
    Natural Language Processing (NLP) is a field of AI that focuses on the interaction between
    computers and human language. It enables machines to understand, interpret, and generate
    human language in a meaningful way.
    """
    
    # Write sample content to a file
    with open("sample_ai_document.txt", "w", encoding="utf-8") as f:
        f.write(sample_content)
    
    upload_result = client.upload_document(
        "sample_ai_document.txt",
        description="Sample document about AI and Machine Learning",
        tags=["artificial-intelligence", "machine-learning", "deep-learning", "nlp"],
        metadata={"category": "technology", "author": "AI Assistant"}
    )
    
    if "error" not in upload_result:
        print(f"Document uploaded successfully!")
        print(f"Document ID: {upload_result.get('document_id')}")
        print(f"Status: {upload_result.get('status')}")
        print(f"Message: {upload_result.get('message')}")
        
        # Wait a bit for processing
        print("Waiting for document processing...")
        time.sleep(3)
    else:
        print(f"Upload failed: {upload_result['error']}")
    
    print()
    
    # 3. Search documents
    print("3. Searching documents...")
    search_queries = [
        "What is artificial intelligence?",
        "How does machine learning work?",
        "What is deep learning used for?",
        "Explain natural language processing"
    ]
    
    for query in search_queries:
        print(f"\nSearching for: '{query}'")
        search_result = client.search_documents(query, max_results=3)
        
        if "error" not in search_result:
            print(f"Found {search_result.get('total_results', 0)} results")
            for i, result in enumerate(search_result.get('results', [])[:2]):
                print(f"  Result {i+1}:")
                print(f"    Content: {result.get('content', '')[:100]}...")
                print(f"    Score: {result.get('score', 'N/A')}")
                print(f"    Filename: {result.get('filename', 'N/A')}")
        else:
            print(f"Search failed: {search_result['error']}")
    
    print()
    
    # 4. Chat with documents
    print("4. Chatting with documents...")
    chat_questions = [
        "What are the main types of AI technologies?",
        "How does machine learning differ from traditional programming?",
        "What is the relationship between deep learning and neural networks?"
    ]
    
    chat_history = []
    for question in chat_questions:
        print(f"\nQuestion: {question}")
        chat_result = client.chat_with_documents(question, chat_history)
        
        if "error" not in chat_result:
            answer = chat_result.get('answer', 'No answer received')
            sources = chat_result.get('sources', [])
            processing_time = chat_result.get('processing_time', 0)
            
            print(f"Answer: {answer}")
            print(f"Processing time: {processing_time:.2f} seconds")
            print(f"Sources: {len(sources)} documents referenced")
            
            # Add to chat history
            chat_history.append({"role": "user", "content": question})
            chat_history.append({"role": "assistant", "content": answer})
        else:
            print(f"Chat failed: {chat_result['error']}")
    
    print()
    
    # 5. List documents
    print("5. Listing documents...")
    documents = client.list_documents()
    if documents and "error" not in documents[0]:
        print(f"Found {len(documents)} documents:")
        for doc in documents:
            print(f"  - {doc.get('filename', 'Unknown')} (ID: {doc.get('document_id', 'N/A')})")
    else:
        print("No documents found or error occurred")
    
    print("\n=== Example completed ===")

if __name__ == "__main__":
    main() 