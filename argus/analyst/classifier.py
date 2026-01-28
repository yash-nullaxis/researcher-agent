from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from ..llm import get_llm
from ..config import AgentConfig

class ComplexityResult(BaseModel):
    is_complex: bool = Field(description="True if the query is complex and requires decomposition to multiple steps, False if it is simple and can be answered with a single SQL query.")
    reasoning: str = Field(description="The reasoning behind the complexity classification.")

class QueryClassifier:
    def __init__(self, config: AgentConfig):
        self.config = config
        self.llm = get_llm(config.model)

    def classify(self, query: str) -> ComplexityResult:
        parser = JsonOutputParser(pydantic_object=ComplexityResult)
        
        system_prompt = (
            "You are a smart assistant that classifies user queries for a data analysis agent.\n"
            "Your job is to determine if a query is 'Simple' (can be answered with a single direct SQL query) "
            "or 'Complex' (requires decomposition, multiple steps, finding multiple data points, or data profiling/reasoning).\n\n"
            "Examples of Simple Queries:\n"
            "- 'How many users are there?'\n"
            "- 'Show me the top 5 products by sales.'\n"
            "- 'List all flights in 2025.'\n"
            "- 'What is the average price of items?'\n\n"
            "Examples of Complex Queries:\n"
            "- 'Compare the sales performance of region A vs region B and explain the trend.'\n"
            "- 'Find anomalies in user signups and correlated events.'\n"
            "- 'Why did revenue drop last month?'\n"
            "- 'Get me the users who bought X and then Y within 3 days.'\n"
            "- 'Analyze the retention rate by cohort.'\n"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "Query: {query}\n\n{format_instructions}"),
        ])
        
        try:
            chain = prompt | self.llm | parser
            result = chain.invoke({
                "query": query,
                "format_instructions": parser.get_format_instructions()
            })
            return ComplexityResult(**result)
        except Exception as e:
            # Fallback to true (complex) to be safe if classification fails
            return ComplexityResult(is_complex=True, reasoning=f"Error during classification: {e}")
