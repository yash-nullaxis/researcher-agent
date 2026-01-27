import asyncio
import os
from argus import Orchestrator, AgentConfig, ModelConfig, DuckDBConnector, SqlAlchemyConnector
from pydantic import SecretStr

# Setup specialized dbs
def setup_dbs():
    import duckdb
    # 1. SQL Source (DuckDB pretending to be Postgres)
    conn_sql = duckdb.connect("sales.db")
    conn_sql.execute("CREATE TABLE IF NOT EXISTS orders (id INTEGER, amount INTEGER, date DATE)")
    conn_sql.execute("INSERT INTO orders VALUES (1, 100, '2024-01-01'), (2, 200, '2024-01-02')")
    conn_sql.close()
    
    # 2. File Source (CSV)
    with open("customers.csv", "w") as f:
        f.write("id,name,segment\n1,Alice,VIP\n2,Bob,Regular")

async def main():
    api_key = os.getenv("OPENAI_API_KEY", "sk-dummy")
    config = AgentConfig(
        model=ModelConfig(provider="openai", model_name="gpt-4o", api_key=SecretStr(api_key)),
        db_connection_str="sqlite:///:memory:" # Ignored
    )
    
    # Initialize Connectors
    # Source 1: "sales_db"
    sales_conn = SqlAlchemyConnector("duckdb:///sales.db")
    
    # Source 2: "crm_files"
    crm_conn = DuckDBConnector(files=["customers.csv"])
    
    # Registry
    connectors = {
        "sales_db": sales_conn,
        "crm_files": crm_conn
    }
    
    agent = Orchestrator(config, connectors=connectors)
    
    print("\n--- Multi-Source Schema ---")
    print(agent.inspector.get_summary())
    
    print("\n--- Running Fusion Query ---")
    # This query requires fusion: Join sales (SQL) and customers (File)
    # The agent should decompose:
    # 1. Get high value orders from sales_db
    # 2. Get VIP customers from crm_files
    # 3. Compare/Fuse (Synthesizer or internal python step)
    # Note: Currently Argus does Synthesis join, not SQL join across connectors (federated SQL).
    # It performs steps sequentially and synthesizer aggregates.
    
    await agent.run("Analyze total sales for VIP customers")
    print("Multi-source run complete.")

if __name__ == "__main__":
    setup_dbs()
    asyncio.run(main())
