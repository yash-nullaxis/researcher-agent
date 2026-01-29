# Argus SDK

Autonomous Business Intelligence Agent for multi-source data connectivity, analysis, and synthesis. Argus empowers users to query databases using natural language, automatically decomposing complex questions into executable SQL queries, validating results, and synthesizing comprehensive reports.

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
    *   Analyzes the user's natural language request and the available database schema.
    *   Generates a list of analysis steps to answer the original question.

3.  **Analyst (`argus.analyst.*`)**:
    *   Executes the plan step-by-step.
    *   Contains components for SQL synthesis, validation, execution, and relevance checking.

4.  **Synthesizer (`argus.synthesizer.memo.Synthesizer`)**:
    *   Aggregates results and generates a final report.

---

## 🛠️ Installation

```bash
pip install .
```

---

## ⚙️ Configuration

Argus is configured via the `AgentConfig` object.

### `AgentConfig` Summary

| Field | Type | Description | Default |
| :--- | :--- | :--- | :--- |
| `model` | `ModelConfig` | LLM configuration (provider, model_name, api_key). | - |
| `db_connection_str` | `str` | Default database connection string. | - |
| `max_steps` | `int` | Maximum analysis steps (limit). | 15 |
| `max_retry_per_step` | `int` | Retries for SQL correction per step. | 3 |
| `verbose` | `bool` | Enable detailed logging. | `False` |

---

## 💻 Usage Example

```python
import asyncio
from argus.config import AgentConfig, ModelConfig
from argus.orchestrator import Orchestrator

async def main():
    config = AgentConfig(
        model=ModelConfig(
            provider="google",
            model_name="gemini-2.0-flash-exp",
            api_key="..."
        ),
        db_connection_str="postgresql://user:password@localhost:5432/my_db"
    )

    orchestrator = Orchestrator(config)
    result = await orchestrator.run("What is the total revenue by region?")
    print(result["final_memo"])

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 📘 Full Documentation

For detailed information on architecture, advanced configurations, and multi-source setups, see the [Full Documentation](./argus/DOCUMENTATION.md).
