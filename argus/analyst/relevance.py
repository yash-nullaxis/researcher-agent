from langchain_core.prompts import ChatPromptTemplate
from ..llm import get_llm
from ..config import AgentConfig
from pydantic import BaseModel, Field

class RelevanceScore(BaseModel):
    is_relevant: bool = Field(description="Whether the result answers the specific part of the user query.")
    score: int = Field(description="Relevance score from 0 to 10.")
    reasoning: str = Field(description="Why this result is relevant or not.")

class RelevanceChecker:
    def __init__(self, config: AgentConfig):
        self.llm = get_llm(config.model)
    
    def check_relevance(self, query: str, goal: str, result_summary: str) -> RelevanceScore:
        """
         checks if the SQL result is relevant to the step goal.
        """
        structured_llm = self.llm.with_structured_output(RelevanceScore)
        
        system_prompt = """You are a Quality Assurance Analyst. Verify if the data analysis result matches the user's intent.
        
        Rate relevance from 0-10.
        If the result is empty or error, relevance is likely 0.
        If the result answers the question, relevance is high.
        """
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "Goal: {goal}\nQuery Executed: {query}\nResult Summary: {result}")
        ])
        
        chain = prompt | structured_llm
        return chain.invoke({"goal": goal, "query": query, "result": result_summary})
