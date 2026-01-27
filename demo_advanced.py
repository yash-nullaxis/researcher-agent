import asyncio
import os
from argus import Orchestrator, AgentConfig, ModelConfig, DuckDBConnector, MongoDBConnector, ChromaDBConnector
from pydantic import SecretStr

async def main():
    api_key = os.getenv("OPENAI_API_KEY", "sk-dummy")
    config = AgentConfig(
        model=ModelConfig(provider="openai", model_name="gpt-4o", api_key=SecretStr(api_key)),
        db_connection_str="sqlite:///:memory:" # Ignored if connector is passed
    )

    print("--- 1. XLSX Demo (DuckDB) ---")
    # Assuming data/sales.xlsx exists
    # connector = DuckDBConnector(files=["data/sales.xlsx"])
    # agent = Orchestrator(config, connector=connector)
    # await agent.run("Analyze sales trends")
    print("Code ready for XLSX.")

    print("\n--- 2. MongoDB Demo ---")
    # connector = MongoDBConnector("mongodb://localhost:27017", "analytics_db")
    # agent = Orchestrator(config, connector=connector)
    # await agent.run("Find users with age > 25")
    print("Code ready for MongoDB.")

    print("\n--- 3. ChromaDB Demo ---")
    # connector = ChromaDBConnector(collection_name="financial_docs")
    # agent = Orchestrator(config, connector=connector)
    # await agent.run("Summarize the Q3 report")
    print("Code ready for ChromaDB.")

if __name__ == "__main__":
    asyncio.run(main())
