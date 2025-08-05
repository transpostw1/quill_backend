from langchain_qdrant import QdrantVectorStore
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.prompts import ChatPromptTemplate
import pyodbc

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

def get_user_query():
    return input("💬 Ask your question about the data: ")

def search_schema(qdrant, embeddings, query, k=5):
    print("\n🔍 Searching schema context from Qdrant...")
    docs = qdrant.similarity_search(query, k=k)
    return "\n\n".join([doc.page_content for doc in docs])

def generate_sql(llm, context, question):
    print("🤖 Generating SQL using LLM (mistral)...")
    prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant that generates valid SQL Server T-SQL queries based on the question and schema context.

IMPORTANT: Use the correct database format: [SISL Live].[dbo].[TableName]
The database has tables like SISL$Customer, SISL$Company, etc.

SCHEMA INFORMATION:
{context}

USER QUESTION:
{question}

Only output the SQL query. No explanations. Use the correct table names from the schema information.
""")
    chain = prompt | llm
    result = chain.invoke({"context": context, "question": question})
    return result.content.strip()

def run_sql_query(conn_str, sql_query):
    print("\n🧠 Executing SQL query on MSSQL...")
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()
    cursor.execute(sql_query)
    
    columns = [column[0] for column in cursor.description]
    rows = cursor.fetchall()
    
    print("\n📊 Query Results:")
    print(columns)
    for row in rows:
        print(row)

if __name__ == "__main__":
    try:
        # Embedding & Vector DB
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

        # Chat LLM
        llm = ChatOllama(
            model="mistral:latest", 
            temperature=0.1,
            base_url="http://192.168.0.120:11434"
        )

        # Pipeline
        user_question = get_user_query()
        context = search_schema(qdrant, embeddings, user_question)
        generated_sql = generate_sql(llm, context, user_question)

        print("\n📄 Generated SQL:\n", generated_sql)
        run_sql_query(MSSQL_CONN_STR, generated_sql)

    except Exception as e:
        print("\n❌ ERROR:\n", e)
