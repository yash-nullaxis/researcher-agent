import asyncio
import os
import duckdb
from pydantic import SecretStr
from argus import Orchestrator, AgentConfig, ModelConfig

# Setup a dummy database
def setup_db():
    conn = duckdb.connect("sample.db")
    conn.execute("CREATE TABLE IF NOT EXISTS sales (id INTEGER, product VARCHAR, amount INTEGER, region VARCHAR)")
    conn.execute("INSERT INTO sales VALUES (1, 'Widget A', 100, 'North'), (2, 'Widget B', 200, 'South'), (3, 'Widget A', 150, 'South')")
    conn.close()
    print("Database 'sample.db' created with table 'sales'.")

async def main():
    # Ensure API Key is present
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set OPENAI_API_KEY environment variable.")
        return

    # Configuration
    config = AgentConfig(
        model=ModelConfig(
            provider="openai",
            model_name="gpt-4o",
            api_key=SecretStr(api_key)
        ),
        db_connection_str="duckdb:///sample.db"
    )

    # Initialize Agent
    agent = Orchestrator(config)

    # Run Query
    query = "What is the total sales amount for Widget A?"
    print(f"User Query: {query}")
    print("Agent running...")
    
    result = await agent.run(query)
    
    print("\n--- Final Result ---")
    print(result['final_memo'])

if __name__ == "__main__":
    setup_db()
    asyncio.run(main())
