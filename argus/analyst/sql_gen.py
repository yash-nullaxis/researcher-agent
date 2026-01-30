import logging
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from ..config import AgentConfig
from ..llm import get_llm

logger = logging.getLogger(__name__)


class SQLQuery(BaseModel):
    query: str = Field(description="Syntactically correct SQL query")
    explanation: str = Field(description="Explanation of what the query does")


class SQLSynthesizer:
    def __init__(self, config: AgentConfig):
        self.config = config
        self.llm = get_llm(config.model)

    def generate_sql(
        self,
        goal: str,
        schema_context: str,
        dialect: str = "duckdb",
        error_context: str | None = None,
    ) -> str:
        from langchain_core.output_parsers import JsonOutputParser

        parser = JsonOutputParser(pydantic_object=SQLQuery)

        # Dialect‑aware guidance and examples (kept short for token efficiency).
        dialect_hint = ""
        if dialect.lower().startswith("duck"):
            dialect_hint = (
                "Use DuckDB‑compatible syntax. Example: "
                "SELECT date_trunc('day', ts) AS day, COUNT(*) FROM events "
                "GROUP BY day ORDER BY day;"
            )
        elif dialect.lower().startswith("post"):
            dialect_hint = (
                "Use PostgreSQL‑compatible syntax. Example: "
                "SELECT date_trunc('day', ts) AS day, AVG(amount) FROM payments "
                "GROUP BY day ORDER BY day;"
            )

        base_prompt = (
            f"You are an expert {dialect} SQL writer. Write a single SELECT query to "
            "accomplish the user's goal.\n\n"
            "Rules:\n"
            "- Use the specific column names and table names from the schema.\n"
            "- Do not hallucinate columns or tables.\n"
            "- If the goal requires multiple steps efficiently expressible in one query "
            "(e.g. CTEs), you may use them.\n"
            "- Always include an explicit LIMIT if the goal does not require full-table scans.\n"
            "- If the schema clearly cannot answer the question, write a harmless query "
            "that returns zero rows (for example, WHERE 1 = 0) rather than inventing tables.\n"
            "- Return only the SELECT statement.\n"
            "- Respond with a valid JSON object containing:\n"
            '  - \"query\": The SQL string.\n'
            '  - \"explanation\": Brief explanation.\n'
        )

        if dialect_hint:
            base_prompt += "\n\nDialect guidance:\n" + dialect_hint + "\n"

        if error_context:
            base_prompt += (
                "\nCRITICAL: Your previous attempt failed with this error:\n"
                f"{error_context}\n"
                "Carefully fix your SQL logic or syntax so that the new query avoids this error."
            )

        system_prompt = base_prompt

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                (
                    "user",
                    "Schema:\n{schema}\n\nGoal: {goal}\n\n{format_instructions}",
                ),
            ]
        )

        chain = prompt | self.llm | parser
        try:
            result = chain.invoke(
                {
                    "schema": schema_context,
                    "goal": goal,
                    "format_instructions": parser.get_format_instructions(),
                }
            )
            # Handle potential casing issues or extra keys
            return result.get("query") or result.get("sql") or ""
        except Exception as e:
            # Fallback text parsing if JSON fails (Gemini sometimes returns markdown json)
            logger.error(f"SQL Gen Error: {e}")
            return ""
