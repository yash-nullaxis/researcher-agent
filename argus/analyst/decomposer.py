import logging
from typing import Dict, List
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from ..llm import get_llm
from ..config import AgentConfig
from ..state import AnalysisStep

logger = logging.getLogger(__name__)


class AnalysisPlan(BaseModel):
    steps: List[AnalysisStep] = Field(description="List of analysis steps to perform")

class QueryDecomposer:
    def __init__(self, config: AgentConfig):
        self.config = config
        self.llm = get_llm(config.model)
    
    def decompose(
        self,
        query: str,
        context: str = "",
        datasources: Dict[str, str] | None = None,
    ) -> List[AnalysisStep]:
        """
        Decomposes a user query into specific analysis steps.
        Now includes 'context' (schema info) to help the planner make better decisions.
        """
        from langchain_core.output_parsers import JsonOutputParser
        
        parser = JsonOutputParser(pydantic_object=AnalysisPlan)

        ds_description = ""
        if datasources:
            parts = []
            for name, hint in datasources.items():
                label = f"- {name}"
                if hint:
                    label += f": {hint}"
                parts.append(label)
            ds_description = "\nYou have access to the following data sources:\n" + "\n".join(parts)
        
        system_prompt = (
            "You are a senior data analyst. Your goal is to create a step-by-step "
            "execution plan to answer the user's business question.\n\n"
            "Capabilities:\n"
            "- You have access to multiple data sources (SQL, Vector, Files).\n"
            "- You can execute SQL queries.\n"
            "- You can perform final synthesis.\n\n"
            "Available Tools:\n"
            "- sql: Generate and execute a SQL query.\n"
            "- python: (Not yet implemented, generally avoid unless necessary).\n\n"
            "Rules:\n"
            "- Break down complex 'compare' questions into separate data retrieval steps.\n"
            "- Prefer a small number of high‑quality steps (typically 3‑7).\n"
            "- Ensure steps are logical and sequential. Assign unique IDs to steps.\n"
            "- If a step requires the result of a previous step (e.g. 'Compare X and Y' "
            "needs 'Get X'), set 'dependency' to the ID of that previous step.\n"
            "- Use the provided schema context to infer feasibility (e.g. if table exists).\n"
            "- IMPORTANT: For each step, choose the most appropriate 'datasource' "
            "from the list of available data sources. If unsure, use 'default'.\n\n"
            "Output Format:\n"
            "Return a valid JSON object with a single key 'steps' containing a list of "
            "step objects. Each step object must have: id, description, tool, "
            "datasource, dependency, thought.\n"
        )

        if ds_description:
            system_prompt += "\n" + ds_description + "\n"
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            (
                "user",
                "Schema Context:\n{context}\n\nUser Question: {query}\n\n"
                "{format_instructions}",
            ),
        ])
        
        chain = prompt | self.llm | parser
        
        # Retry logic could be added here, but for now we rely on the parser
        try:
            result = chain.invoke(
                {
                    "query": query,
                    "context": context,
                    "format_instructions": parser.get_format_instructions(),
                }
            )
            # Result is a dict, likely {'steps': [...]}
            # Validate with Pydantic
            plan = AnalysisPlan(**result)
            return plan.steps
        except Exception as e:
            # Fallback or simple re-raise
            logger.error(f"Decomposition Error: {e}")
            return []
