from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from ..config import AgentConfig
from ..llm import get_llm
from ..state import AnalysisState


class Synthesizer:
    def __init__(self, config: AgentConfig):
        self.config = config
        self.llm = get_llm(config.model)

    def synthesize(self, state: AnalysisState) -> str:
        """
        Generates a final answer (Executive Memo) based on all step results.
        """
        results_summary = ""
        plan = state.get("plan", []) or []
        # Create a map for quick lookup
        step_map = {step.id: step for step in plan}

        for res in state.get("step_results", []):
            step = step_map.get(res.step_id)
            step_desc = step.description if step is not None else "Unknown Step"
            results_summary += f"Step {res.step_id}: {step_desc}\n"
            if res.query_executed:
                results_summary += f"Query: {res.query_executed}\n"
            if res.success:
                results_summary += f"Result: {res.output}\n\n"
            else:
                results_summary += f"Error: {res.error}\n\n"

        # Optionally append a trace appendix when verbose
        debug_lines = []
        if self.config.verbose and state.get("debug_trace"):
            debug_lines.append("\n\nAppendix: Execution Trace\n")
            for entry in state["debug_trace"]:
                debug_lines.append(f"- {entry}")

        if debug_lines:
            results_with_trace = results_summary + "\n".join(debug_lines)
        else:
            results_with_trace = results_summary

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an Executive Strategy Consultant. Write a final memo answering the user's original question based on the analysis results.

Structure:
1. Executive Summary (BLUF)
2. Key Findings (each finding should be self-contained and richly detailed, without phrases like "see Step 1/2/3"; repeat the relevant numbers and context inline)
3. Recommended Actions

Format: Markdown.
""",
                ),
                (
                    "user",
                    "User Question: {question}\n\nAnalysis Results:\n{results}",
                ),
            ]
        )

        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke(
            {"question": state["user_query"], "results": results_with_trace}
        )
