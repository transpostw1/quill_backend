from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uvicorn
from langchain_qdrant import QdrantVectorStore
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage
import pyodbc
import logging
import threading

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Config ---
QDRANT_URL = "http://192.168.0.120:6333/"
COLLECTION_NAME = "schema_vectors"

# 🔧 Replace these with your actual MSSQL credentials
MSSQL_CONN_STR = (
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=192.168.1.44,1433;"
    "DATABASE=SISL Live;"
    "UID=test;"
    "PWD=Test@345;"
    "TrustServerCertificate=yes;"
)

# Initialize FastAPI app
app = FastAPI(
    title="SISL RAG API",
    description="RAG API for querying SISL database using natural language",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class QueryRequest(BaseModel):
    question: str
    chat_id: Optional[str] = None
    max_results: Optional[int] = 5

class QueryResponse(BaseModel):
    question: str
    generated_sql: str
    results: List[Dict[str, Any]]
    columns: List[str]
    row_count: int
    success: bool
    error: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    message: str

# Global variables for initialized components
qdrant = None
embeddings = None
llm = None

# Memory manager for chat sessions
class ChatMemoryManager:
    def __init__(self):
        self.memories = {}
        self.lock = threading.Lock()
    
    def get_memory(self, chat_id: str) -> ConversationBufferMemory:
        """Get or create conversation memory for a chat ID"""
        with self.lock:
            if chat_id not in self.memories:
                self.memories[chat_id] = ConversationBufferMemory(
                    memory_key="chat_history", 
                    return_messages=True
                )
            return self.memories[chat_id]
    
    def clear_memory(self, chat_id: str):
        """Clear conversation memory for a chat ID"""
        with self.lock:
            if chat_id in self.memories:
                del self.memories[chat_id]
    
    def cleanup_old_memories(self, max_memories: int = 1000):
        """Clean up old memories to prevent memory leaks"""
        with self.lock:
            if len(self.memories) > max_memories:
                # Remove oldest memories (FIFO)
                keys_to_remove = list(self.memories.keys())[:len(self.memories) - max_memories]
                for key in keys_to_remove:
                    del self.memories[key]

# Initialize memory manager
memory_manager = ChatMemoryManager()

def initialize_components():
    """Initialize Qdrant, embeddings, and LLM components"""
    global qdrant, embeddings, llm
    
    try:
        # Initialize embeddings
        embeddings = OllamaEmbeddings(
            model="nomic-embed-text",
            base_url="http://192.168.0.120:11434"
        )
        
        # Parse URL to get host and port
        url_parts = QDRANT_URL.replace("http://", "").replace("https://", "").split(":")
        host = url_parts[0]
        port = int(url_parts[1].replace("/", ""))
        
        from qdrant_client import QdrantClient
        
        client = QdrantClient(host=host, port=port)
        qdrant = QdrantVectorStore(
            collection_name=COLLECTION_NAME,
            embedding=embeddings,
            client=client
        )

        # Initialize LLM
        llm = ChatOllama(
            model="mistral:latest", 
            temperature=0.1,
            base_url="http://192.168.0.120:11434"
        )
        
        logger.info("All components initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        return False

def search_schema(query: str, k: int = 5) -> str:
    """Search schema context from Qdrant"""
    try:
        docs = qdrant.similarity_search(query, k=k)
        return "\n\n".join([doc.page_content for doc in docs])
    except Exception as e:
        logger.error(f"Error searching schema: {e}")
        raise

def generate_sql(context: str, question: str) -> str:
    """Generate SQL using LLM"""
    try:
        prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant that generates valid SQL Server T-SQL queries based on the question and schema context.

IMPORTANT DATABASE STRUCTURE:
- Use the correct database format: [SISL Live].[dbo].[TableName]
- The database has tables with patterns like: ssil_UAT$Customer, ssil_UAT$Sales Invoice Header, ssil_UAT$Sales Invoice Line
- Customer table: ssil_UAT$Customer (columns: No_, Name)
- Sales tables: ssil_UAT$Sales Invoice Header, ssil_UAT$Sales Invoice Line
- Use proper column names: No_ (customer number), Name (customer name), Amount Including VAT (sales amount)

CORRECT EXAMPLES:
- Customer count: SELECT COUNT(*) FROM [SISL Live].[dbo].[ssil_UAT$Customer]
- Top customers by sales: 
  SELECT TOP 5 c.[No_] AS [Customer Number], c.[Name] AS [Customer Name], 
         SUM(sil.[Amount Including VAT]) AS [Total Sales] 
  FROM [SISL Live].[dbo].[ssil_UAT$Customer] c 
  INNER JOIN [SISL Live].[dbo].[ssil_UAT$Sales Invoice Header] sih ON c.[No_] = sih.[Sell-to Customer No_] 
  INNER JOIN [SISL Live].[dbo].[ssil_UAT$Sales Invoice Line] sil ON sih.[No_] = sil.[Document No_] 
  GROUP BY c.[No_], c.[Name] 
  ORDER BY [Total Sales] DESC

SCHEMA INFORMATION:
{context}

USER QUESTION:
{question}

Only output the SQL query. No explanations. Use the correct table names from the examples above.
""")
        chain = prompt | llm
        result = chain.invoke({"context": context, "question": question})
        return result.content.strip()
    except Exception as e:
        logger.error(f"Error generating SQL: {e}")
        raise

def run_sql_query(sql_query: str) -> tuple[List[str], List[tuple], int]:
    """Execute SQL query and return columns, rows, and row count"""
    try:
        conn = pyodbc.connect(MSSQL_CONN_STR)
        cursor = conn.cursor()
        cursor.execute(sql_query)
        
        columns = [column[0] for column in cursor.description]
        rows = cursor.fetchall()
        row_count = len(rows)
        
        cursor.close()
        conn.close()
        
        return columns, rows, row_count
    except Exception as e:
        logger.error(f"Error executing SQL query: {e}")
        raise

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    logger.info("Starting SISL RAG API...")
    if not initialize_components():
        logger.error("Failed to initialize components. API may not work properly.")

@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        message="SISL RAG API is running"
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        message="SISL RAG API is running"
    )

@app.post("/query", response_model=QueryResponse)
async def query_database(request: QueryRequest):
    """Main endpoint for querying the database using natural language"""
    try:
        logger.info(f"Received query: {request.question}")
        
        # Get or create conversation memory for this chat
        chat_id = request.chat_id or "default"
        conversation_memory = memory_manager.get_memory(chat_id)
        
        # Load conversation history
        memory_vars = conversation_memory.load_memory_variables({})
        chat_history = memory_vars.get("chat_history", [])
        
        # Format chat history for prompt
        history_text = ""
        if chat_history:
            history_text = "\nConversation History:\n"
            for msg in chat_history[-6:]:  # Last 3 exchanges (6 messages)
                if isinstance(msg, HumanMessage):
                    history_text += f"User: {msg.content}\n"
                elif isinstance(msg, AIMessage):
                    history_text += f"Assistant: {msg.content}\n"
        
        # Classify intent using LLM with conversation context
        intent_prompt = f"""
        Classify the following user input as either a "database_query" or "general_conversation".
        
        Database queries are questions about data, reports, counts, lists, or information that would require accessing a database.
        Examples of database queries:
        - "How many customers do we have?"
        - "Show me sales by region"
        - "List all products over $100"
        
        General conversation includes greetings, casual chat, or questions that don't require database access.
        Examples of general conversation:
        - "Hi"
        - "How are you?"
        - "What's your name?"
        
        User input: "{request.question}"
{history_text}
        Considering the conversation history, respond with ONLY ONE of these two words: "database_query" or "general_conversation"
        """
        
        intent_result = llm.invoke(intent_prompt)
        intent = intent_result.content.strip().lower()
        
        # If it's general conversation, handle it directly without RAG processing
        if "general_conversation" in intent:
            conversation_prompt = f"""
            You are a helpful assistant for a database querying system. 
            Respond appropriately to this user input: "{request.question}"
            Keep your response concise and friendly.
{history_text}
            """
            
            conversation_result = llm.invoke(conversation_prompt)
            response_text = conversation_result.content.strip()
            
            # Save interaction to memory
            conversation_memory.save_context(
                {"input": request.question}, 
                {"output": response_text}
            )
            
            return QueryResponse(
                question=request.question,
                generated_sql="",
                results=[{"response": response_text}],
                columns=["response"],
                row_count=1,
                success=True
            )
        
        # Search schema context
        context = search_schema(request.question, request.max_results)
        logger.info("Schema context retrieved successfully")
        
        # Generate SQL
        generated_sql = generate_sql(context, request.question)
        logger.info(f"Generated SQL: {generated_sql}")
        
        # Execute SQL query
        columns, rows, row_count = run_sql_query(generated_sql)
        logger.info(f"Query executed successfully. Found {row_count} rows")
        
        # Convert rows to list of dictionaries
        results = []
        for row in rows:
            row_dict = {}
            for i, value in enumerate(row):
                # Convert any non-serializable objects to strings
                if hasattr(value, 'isoformat'):  # datetime objects
                    row_dict[columns[i]] = value.isoformat()
                else:
                    row_dict[columns[i]] = str(value) if value is not None else None
            results.append(row_dict)
        
        # Format results for memory
        results_text = f"Found {row_count} results"
        if results:
            # Include first row as example
            first_row = results[0]
            results_text += f". Example: {first_row}"
        
        # Save interaction to memory
        conversation_memory.save_context(
            {"input": request.question}, 
            {"output": results_text}
        )
        
        return QueryResponse(
            question=request.question,
            generated_sql=generated_sql,
            results=results,
            columns=columns,
            row_count=row_count,
            success=True
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        error_message = f"Error processing query: {str(e)}"
        
        # Save error interaction to memory (if chat_id is available)
        if request.chat_id:
            conversation_memory = memory_manager.get_memory(request.chat_id)
            conversation_memory.save_context(
                {"input": request.question}, 
                {"output": error_message}
            )
        
        return QueryResponse(
            question=request.question,
            generated_sql="",
            results=[],
            columns=[],
            row_count=0,
            success=False,
            error=str(e)
        )

@app.get("/tables")
async def get_tables():
    """Get list of available tables"""
    try:
        sql = "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'BASE TABLE' ORDER BY TABLE_NAME"
        columns, rows, row_count = run_sql_query(sql)
        
        tables = [row[0] for row in rows]
        return {
            "tables": tables,
            "count": len(tables),
            "success": True
        }
    except Exception as e:
        logger.error(f"Error getting tables: {e}")
        return {
            "tables": [],
            "count": 0,
            "success": False,
            "error": str(e)
        }

@app.post("/clear_memory")
async def clear_memory(chat_id: str):
    """Clear conversation memory for a specific chat ID"""
    try:
        memory_manager.clear_memory(chat_id)
        return {"message": f"Memory cleared for chat_id: {chat_id}", "success": True}
    except Exception as e:
        logger.error(f"Error clearing memory for chat_id {chat_id}: {e}")
        return {"message": f"Error clearing memory: {str(e)}", "success": False}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)