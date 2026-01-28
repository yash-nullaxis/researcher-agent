import asyncio
import os

from argus import (
    AgentConfig,
    DuckDBConnector,
    ModelConfig,
    Orchestrator,
    SqlAlchemyConnector,
)
from pydantic import SecretStr


# Setup specialized dbs
def setup_dbs():
    import duckdb

    # 1. SQL Source (DuckDB pretending to be Postgres)
    conn_sql = duckdb.connect("sales.db")
    conn_sql.execute(
        "CREATE TABLE IF NOT EXISTS orders (id INTEGER, amount INTEGER, date DATE)"
    )
    conn_sql.execute(
        "INSERT INTO orders VALUES (1, 100, '2024-01-01'), (2, 200, '2024-01-02')"
    )
    conn_sql.close()

    # 2. File Source (CSV)
    with open("customers.csv", "w") as f:
        f.write("id,name,segment\n1,Alice,VIP\n2,Bob,Regular")


async def main():
    api_key = os.getenv("OPENAI_API_KEY", "sk-dummy")
    config = AgentConfig(
        model=ModelConfig(
            provider="openai", model_name="gpt-4o", api_key=SecretStr(api_key)
        ),
        db_connection_str="sqlite:///:memory:",  # Ignored
        verbose=True,
        log_sql=True,
    )

    # Initialize Connectors
    # Source 1: "sales_db"
    sales_conn = SqlAlchemyConnector("duckdb:///sales.db")

    # Source 2: "crm_files"
    crm_conn = DuckDBConnector(files=["customers.csv"])

    # Registry with explicit hints (used by planner)
    connectors = {
        "sales_db": sales_conn,
        "crm_files": crm_conn,
    }

    agent = Orchestrator(config, connectors=connectors)

    print("\n--- Multi-Source Schema ---")
    print(agent.inspector.get_summary())

    print("\n--- Running Fusion Query ---")
    print("Connectors:", list(connectors.keys()))

    result = await agent.run("Analyze total sales for VIP customers")
    print("\n--- Final Memo ---")
    print(result.get("final_memo", "No memo generated."))


if __name__ == "__main__":
    setup_dbs()
    asyncio.run(main())
