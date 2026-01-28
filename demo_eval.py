import asyncio
import csv
from pathlib import Path
from typing import Dict, Any, List

from argus import Orchestrator, AgentConfig, ModelConfig, SqlAlchemyConnector
from pydantic import SecretStr


EVAL_CSV = Path("eval_questions.csv")


def load_eval_cases() -> List[Dict[str, str]]:
    """
    Load evaluation cases from a simple CSV with columns:
    id,question,notes
    """
    if not EVAL_CSV.exists():
        raise FileNotFoundError(
            f"Evaluation file {EVAL_CSV} not found. "
            "Create it with columns: id,question,notes"
        )

    rows: List[Dict[str, str]] = []
    with EVAL_CSV.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


async def run_eval() -> None:
    """
    Minimal evaluation harness:
    - Reads questions from eval_questions.csv
    - Runs them through the Orchestrator
    - Prints a short, human-readable summary per case
    """
    # This is intentionally simple: adjust DSN/model as needed for your environment.
    dsn = "postgresql://postgres:password@localhost:5432/postgres"
    api_key = "sk-replace-me"

    config = AgentConfig(
        model=ModelConfig(
            provider="openai",
            model_name="gpt-4o-mini",
            api_key=SecretStr(api_key),
        ),
        db_connection_str=dsn,
        verbose=False,
        log_sql=True,
        log_results=False,
    )

    connector = SqlAlchemyConnector(dsn)
    agent = Orchestrator(config, connector=connector)

    cases = load_eval_cases()
    print(f"Loaded {len(cases)} evaluation cases from {EVAL_CSV}")

    for row in cases:
        qid = row.get("id", "").strip()
        question = row.get("question", "").strip()
        notes = row.get("notes", "").strip()

        print("\n" + "=" * 80)
        print(f"[Case {qid}] {question}")
        if notes:
            print(f"Notes: {notes}")

        try:
            result: Dict[str, Any] = await agent.run(question)
        except Exception as e:
            print(f"ERROR running case {qid}: {e}")
            continue

        memo = result.get("final_memo", "")
        print("\nFinal memo (truncated):\n")
        print(memo[:2000])


if __name__ == "__main__":
    asyncio.run(run_eval())

