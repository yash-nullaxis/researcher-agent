import asyncio
import os
import shutil

from argus import (
    AgentConfig,
    ChromaDBConnector,
    ModelConfig,
    Orchestrator,
    SqlAlchemyConnector,
)
from pydantic import SecretStr


def setup_db():
    import duckdb

    conn = duckdb.connect("rag_demo.db")
    # Table 1: Employees (Relevant to "salary")
    conn.execute(
        "CREATE TABLE employees (id INT, name VARCHAR, salary INT, dept_id INT)"
    )
    conn.execute("INSERT INTO employees VALUES (1, 'Alice', 100000, 1)")

    # Table 2: Products (Irrelevant to "salary")
    conn.execute("CREATE TABLE products (sku VARCHAR, price DECIMAL, stock INT)")
    conn.close()


async def main():
    api_key = os.getenv("OPENAI_API_KEY", "sk-dummy")
    config = AgentConfig(
        model=ModelConfig(
            provider="openai", model_name="gpt-4o", api_key=SecretStr(api_key)
        ),
        db_connection_str="sqlite:///:memory:",
        verbose=True,
    )

    # connectors
    sql_conn = SqlAlchemyConnector("duckdb:///rag_demo.db")

    # Vector used strictly for internal metadata (RAG)
    # Using a temp persistence dir
    persist_dir = "./chroma_rag_test"
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)

    import chromadb

    client = chromadb.PersistentClient(path=persist_dir)
    vector_conn = ChromaDBConnector(client=client, collection_name="schema_index")

    connectors = {
        "hr_db": sql_conn,
        "metadata_store": vector_conn,  # Wired to RAG in orchestrator logic
    }

    # Initialize agent
    print("Initializing Agent (Indexing Schemas)...")
    agent = Orchestrator(config, connectors=connectors)

    # Test 1: Check if vector store was populated
    print("\n--- Diagnostic: Vector Store Count ---")
    print(vector_conn.get_schema_info())  # Should show count > 0 if indexing worked

    # Test 2: Run Query
    # Query about "salary" should retrieve 'employees' table context but NOT 'products' ideally (or deprioritize it)
    # The inspector logs "--- RELEVANT TABLES (RAG FILTERED) ---" if logic triggers.
    print("\n--- Running RAG Query ---")

    # We cheat and look at what inspector returns for a query
    summary = agent.inspector.get_summary("Show me employee salaries")
    print("\n[Inspector Output for 'salary' query]:")
    print(summary)

    # Check if we see RELEVANT TABLES header
    if "RELEVANT TABLES" in summary:
        print("\nSUCCESS: RAG Filter activated.")
    else:
        print(
            "\nFAIL: RAG Filter not clearly activated (maybe full summary returned or error)."
        )


if __name__ == "__main__":
    setup_db()
    asyncio.run(main())
