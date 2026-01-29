# Argus SDK Documentation

Argus is an **Autonomous Business Intelligence Agent** designed for multi-source data connectivity, analysis, and synthesis. It empowers users to query databases using natural language, automatically decomposing complex questions into executable SQL queries, validating results, and synthesizing comprehensive reports.

## 🚀 Key Features

*   **Multi-Source Fusion**: Connect to multiple databases (SQL, NoSQL, Vector DB) simultaneously.
*   **LLM-Driven Planning**: Automatically decomposes complex user queries into step-by-step analysis plans.
*   **Self-Healing & Auto-Repair**: Automatically detects SQL errors or irrelevant results and attempts to fix the query.
*   **Schema-Aware**: Uses schema inspection (and optional RAG context) to generate accurate SQL.
*   **Safety & Validation**: Validates generated SQL using `sqlglot` before execution to prevent errors and ensure safety.
*   **Recursive Analysis**: Can iterate through multiple steps of analysis, handling dependencies between steps.

---

## 🏗️ Architecture

Argus follows a **Planner-Analyst-Synthesizer** architecture orchestrated by a state graph (powered by `LangGraph`).

### Core Components

1.  **Orchestrator (`argus.orchestrator.Orchestrator`)**:
    *   The central controller that manages the workflow.
    *   Initializes the services (Planner, Analyst, Synthesizer) and compiling the execution graph.
    *   Manages the global `AnalysisState`.

2.  **Planner (`argus.analyst.decomposer.QueryDecomposer`)**:
    *   **Role**: Analyzes the user's natural language request and the available database schema.
    *   **Output**: A list of `AnalysisStep` objects (the "Plan"). Each step represents a distinct operation (e.g., retrieving data via SQL).
    *   **Context**: Uses `SchemaInspector` to understand table structures.

3.  **Analyst (`argus.analyst.*`)**:
    *   executes the plan step-by-step.
    *   **`SQLSynthesizer`**: Generates SQL queries based on the step description and schema.
    *   **`SQLValidator`**: locally validates SQL syntax and semantics using `sqlglot`.
    *   **`SQLExecutor`**: Executes the valid SQL against the target database.
    *   **`RelevanceChecker`**: Verifies if the query results are relevant to the step's goal.
    *   **Auto-Repair**: If validation or execution fails, the Analyst feeds the error back to the Synthesizer for self-correction (up to a configured retry limit).

4.  **Synthesizer (`argus.synthesizer.memo.Synthesizer`)**:
    *   Aggregates the results from all successful analysis steps.
    *   Generates a final, human-readable memo or report answering the original user query.

---

## 🛠️ Installation & Setup

### Requirements
*   Python 3.10+
*   Dependencies listed in `pyproject.toml` (or `setup.py`)

### Installation

```bash
pip install .
```

---

## ⚙️ Configuration

Argus is highly configurable via the `AgentConfig` object.

### `ModelConfig`
Configuration for the LLM backend (e.g., OpenAI, Gemini, Anthropic).

| Field | Type | Description | Default |
| :--- | :--- | :--- | :--- |
| `provider` | `str` | LLM provider name (e.g., "openai", "google"). | "openai" |
| `model_name` | `str` | Model identifier (e.g., "gpt-4-turbo"). | "gpt-4-turbo" |
| `api_key` | `SecretStr` | API Key for the provider. | - |
| `temperature` | `float` | Model creativity (0.0 for deterministic). | 0.0 |

### `AgentConfig`
Top-level agent settings.

| Field | Type | Description | Default |
| :--- | :--- | :--- | :--- |
| `model` | `ModelConfig` | The LLM configuration. | - |
| `db_connection_str` | `str` | Default database connection string (SQLAlchemy format). | - |
| `max_steps` | `int` | Maximum number of analysis steps allowed. | 15 |
| `recursion_limit` | `int` | Safety limit for total graph parsings. | 25 |
| `max_retry_per_step` | `int` | Retries for SQL correction per step. | 3 |
| `verbose` | `bool` | Enable detailed logging of the agent's thoughts. | `False` |

---

## 💻 Usage Example

Here is a basic example of how to initialize and run the Argus agent against a PostgreSQL database.

```python
import asyncio
import os
from pydantic import SecretStr

from argus.config import AgentConfig, ModelConfig
from argus.orchestrator import Orchestrator
from argus.db.connector import SqlAlchemyConnector

async def main():
    # 1. Configure the Agent
    config = AgentConfig(
        model=ModelConfig(
            provider="google",
            model_name="gemini-2.0-flash-exp",
            api_key=SecretStr(os.environ["GEMINI_API_KEY"]),
            temperature=0
        ),
        db_connection_str="postgresql://user:password@localhost:5432/my_db",
        verbose=True,
        max_retry_per_step=3
    )

    # 2. Setup Database Connector (Optional explicit setup)
    # The Orchestrator can auto-init a default connector from config.db_connection_str
    # but providing one explicitly allows for more control.
    connector = SqlAlchemyConnector(config.db_connection_str)

    # 3. Initialize Orchestrator
    orchestrator = Orchestrator(config, connector=connector)

    # 4. Run Analysis
    user_query = "What is the total revenue by region for the last quarter?"
    print(f"User Query: {user_query}")
    
    result = await orchestrator.run(user_query)
    
    # 5. Output Result
    print("\n=== Final Memo ===\n")
    print(result["final_memo"])

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 🧩 Advanced Usage

### Connecting Multiple Data Sources

Argus can route queries to specific databases if multiple are provided.

```python
from argus.db.connector import SqlAlchemyConnector, DuckDBConnector

# ... config setup ...

pg_connector = SqlAlchemyConnector("postgresql://...")
duck_connector = DuckDBConnector("local_warehouse.db")

connectors = {
    "sales_db": pg_connector,
    "analytics_db": duck_connector
}

# The agent's Planner will decide which source to query based on hints or schema
orchestrator = Orchestrator(config, connectors=connectors)
```

### Handling SQL Dialects

Argus automatically infers the SQL dialect (Postgres, DuckDB, SQLite) based on the connector engine. This ensures that the `SQLSynthesizer` generates dialect-specific syntax (e.g., date functions, window functions).

### Auto-Repair Logic

If a generated SQL query fails execution (e.g., syntax error, non-existent column) or returns invalid/irrelevant data:
1.  The error is captured by the `analyst_node`.
2.  The workflow cycles back to the `analyst` node (instead of moving to the next step).
3.  The `SQLSynthesizer` is re-invoked with the **original request**, **schema**, and the **Error Message**.
4.  The LLM attempts to correct the SQL.
5.  This repeats until success or `max_retry_per_step` is reached.
