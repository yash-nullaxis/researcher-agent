import asyncio
import os
import argparse
from dotenv import load_dotenv
from argus import Orchestrator, AgentConfig, ModelConfig, SqlAlchemyConnector
from pydantic import SecretStr

# Load .env to get GOOGLE_API_KEY
load_dotenv()

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true", help="Enable aggressive logging")
    args = parser.parse_args()

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY not found in environment.")
        return

    # 1. Configuration for Gemini 2.0 Flash
    config = AgentConfig(
        model=ModelConfig(
            provider="google", 
            model_name="gemini-2.0-flash", 
            api_key=SecretStr(api_key)
        ),
        # This DSN is passed here but also used explicitly in connector below
        db_connection_str="postgresql://postgres:0h5UPFxhRWUFwdwE@localhost:5432/postgres", 
        verbose=args.verbose
    )

    # 2. Setup Data Connector for Postgres
    postgres_dsn = "postgresql://postgres:0h5UPFxhRWUFwdwE@localhost:5432/postgres"
    print(f"Connecting to Postgres: {postgres_dsn}...")
    
    try:
        connector = SqlAlchemyConnector(connection_str=postgres_dsn)
        # Test connection structure
        info = connector.get_schema_info()
        print(f"Connected! Schema Info Preview:\n{info[:500]}...")
    except Exception as e:
        print(f"Failed to connect to Postgres: {e}")
        print("Ensure you have 'psycopg2-binary' installed: pip install psycopg2-binary")
        return

    # 3. Initialize Agent
    agent = Orchestrator(config, connector=connector)

    # 4. Run Analysis
    # A generic query that works on any schema
    query = "commpare the block time vs actual time of indigo and ix"
    
    print(f"\n--- Running Postgres Verification with Gemini 2.0 Flash ---\nQuery: {query}")
    result = await agent.run(query)
    
    print("\n\n--- Final Memo ---")
    print(result.get("final_memo", "No memo generated."))

if __name__ == "__main__":
    asyncio.run(main())
