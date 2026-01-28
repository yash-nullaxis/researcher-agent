from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from ..config import AgentConfig
from ..llm import get_llm


class RelevanceScore(BaseModel):
    is_relevant: bool = Field(
        description="Whether the result answers the specific part of the user query."
    )
    score: int = Field(description="Relevance score from 0 to 10.")
    reasoning: str = Field(description="Why this result is relevant or not.")


class RelevanceChecker:
    def __init__(self, config: AgentConfig):
        self.config = config
        self.llm = get_llm(config.model)

    def check_relevance(
        self, query: str, goal: str, result_summary: str
    ) -> RelevanceScore:
        """
        Checks if the SQL result is relevant to the step goal.
        """
        structured_llm = self.llm.with_structured_output(RelevanceScore)

        system_prompt = """
You are a Quality Assurance Analyst for data analysis.

Evaluate whether the SQL result actually answers the analytic goal.

Guidelines:
- Rate relevance from 0-10.
- If the result is empty, clearly off-topic, or aggregates the wrong metric
  (for example, counts rows when the goal asks for an average over time),
  the relevance should be low (0-3).
- If the result partially answers the goal but misses key constraints
  (wrong time grain, missing filters, wrong grouping), score it in the mid range (4-6).
- If the result directly and correctly answers the goal using appropriate filters,
  aggregations, and groupings, relevance should be high (7-10).
"""

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                (
                    "user",
                    "Goal: {goal}\nQuery Executed: {query}\nResult Summary: {result}",
                ),
            ]
        )

        chain = prompt | structured_llm
        return chain.invoke({"goal": goal, "query": query, "result": result_summary})
