import asyncio
import os
from dotenv import load_dotenv
from argus import Orchestrator, AgentConfig, ModelConfig, DuckDBConnector
from pydantic import SecretStr

# Load .env to get GOOGLE_API_KEY
load_dotenv()

async def main():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY not found in environment.")
        return

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true", help="Enable aggressive logging")
    args = parser.parse_args()

    # 1. Configuration for Gemini 2.0 Flash
    config = AgentConfig(
        model=ModelConfig(
            provider="google", 
            model_name="gemini-2.0-flash", 
            api_key=SecretStr(api_key)
        ),
        db_connection_str="sqlite:///:memory:",
        verbose=args.verbose
    )

    # 2. Setup Data Connector with User Files
    # Note: DuckDB auto-detects CSV structure. 
    # For XLSX, we rely on the pandas fallback we implemented in DuckDBConnector.
    user_files = [
        "/Users/yashsomkuwar/Documents/researcher-agent/file_example_XLSX_100.xlsx",
        "/Users/yashsomkuwar/Documents/researcher-agent/IX_Flown_Converted_New.csv"
    ]
    
    print(f"Loading files: {user_files}...")
    try:
        connector = DuckDBConnector(files=user_files)
    except Exception as e:
        print(f"Failed to load files: {e}")
        return

    # 3. Initialize Agent
    agent = Orchestrator(config, connector=connector)

    # 4. Run Analysis
    # We ask a broad question to test schema inspection and data retrieval
    query = "Analyze the flight data (IX_Flown) to find the most delayed flight, and check the Excel file for any correlating user data (if applicable)."
    
    print(f"\n--- Running Verification Query with Gemini 2.0 Flash ---\nQuery: {query}")
    result = await agent.run(query)
    
    print("\n\n--- Final Memo ---")
    print(result.get("final_memo", "No memo generated."))

if __name__ == "__main__":
    asyncio.run(main())
