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
        verbose=args.verbose,
        log_sql=True,
    )

    # 2. Setup Data Connector for Postgres
    postgres_dsn = "postgresql://postgres:0h5UPFxhRWUFwdwE@localhost:5432/postgres"
    print(f"Connecting to Postgres: {postgres_dsn}...")
    
    try:
        connector = SqlAlchemyConnector(connection_str=postgres_dsn)
        
        # Detailed connectivity check
        print("Verifying connection and fetching schema...")
        info = connector.get_schema_info()
        
        # Attempt to get precise table count
        try:
            schema_dict = connector.get_schema_dict()
            table_count = len(schema_dict)
            print(f"Connection Successful! Found {table_count} tables.")
        except Exception:
            print("Connection Successful! (Could not enumerate tables details)")

        print(f"Schema Info Preview:\n{info[:500]}...")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to connect to Postgres: {e}")
        print("Ensure you have 'psycopg2-binary' installed: pip install psycopg2-binary")
        print("Check your database URL and ensure the server is running.")
        return

    # 3. Initialize Agent
    agent = Orchestrator(config, connector=connector)

    # 4. Run Analysis
    # A generic query that works on any schema
    query = """
    Show me the total number of flights for each month in 2025.
    """
    
    print(f"\n--- Running Postgres Verification with Gemini 2.0 Flash ---\nQuery: {query}")
    try:
        result = await agent.run(query)
        
        print("\n\n--- Final Memo ---")
        print(result.get("final_memo", "No memo generated."))
    except Exception as e:
        print(f"\n[Error] Agent execution failed during analysis: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
