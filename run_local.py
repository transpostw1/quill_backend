#!/usr/bin/env python3
"""
Script to run the RAG API locally for testing
"""
import uvicorn
import sys
import os

if __name__ == "__main__":
    print("🚀 Starting RAG API locally for testing...")
    print("📍 API will be available at: http://localhost:8000")
    print("📋 Health endpoint: http://localhost:8000/health")
    print("💬 Query endpoint: http://localhost:8000/query")
    print("⏹️  Press CTRL+C to stop the server")
    print("=" * 50)
    
    try:
        # Import the app from api.py
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from api import app
        
        # Run the server
        uvicorn.run(
            "api:app", 
            host="localhost", 
            port=8000, 
            reload=False,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)