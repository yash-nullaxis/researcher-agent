from langchain_core.prompts import ChatPromptTemplate
from ..llm import get_llm
from ..config import AgentConfig
from ..state import AnalysisState
from langchain_core.output_parsers import StrOutputParser

class Synthesizer:
    def __init__(self, config: AgentConfig):
        self.llm = get_llm(config.model)
    
    def synthesize(self, state: AnalysisState) -> str:
        """
        Generates a final answer (Executive Memo) based on all step results.
        """
        results_summary = ""
        # Create a map for quick lookup
        step_map = {step.id: step for step in state['plan']}
        
        for res in state['step_results']:
            # Safe lookup
            step_desc = step_map.get(res.step_id).description if res.step_id in step_map else "Unknown Step"
            results_summary += f"Step {res.step_id}: {step_desc}\n"
            results_summary += f"Query: {res.query_executed}\n"
            if res.success:
                results_summary += f"Result: {res.output}\n\n"
            else:
                results_summary += f"Error: {res.error}\n\n"
            
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an Executive Strategy Consultant. Write a final memo answering the user's original question based on the analysis results.
            
            Structure:
            1. Executive Summary (BLUF)
            2. Key Findings (with data citations)
            3. Recommended Actions
            
            Format: Markdown.
            """),
            ("user", "User Question: {question}\n\nAnalysis Results:\n{results}")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"question": state['user_query'], "results": results_summary})
