from langchain_core.prompts import ChatPromptTemplate
from ..llm import get_llm
from ..config import AgentConfig
from pydantic import BaseModel, Field

class SQLQuery(BaseModel):
    query: str = Field(description="Syntactically correct SQL query")
    explanation: str = Field(description="Explanation of what the query does")

class SQLSynthesizer:
    def __init__(self, config: AgentConfig):
        self.llm = get_llm(config.model)
    
    def generate_sql(self, goal: str, schema_context: str, dialect: str = "duckdb") -> str:
        from langchain_core.output_parsers import JsonOutputParser
        
        parser = JsonOutputParser(pydantic_object=SQLQuery)
        
        system_prompt = f"""You are an expert SQL writer. Write a {dialect} SQL query to accomplish the user's goal.
        
        Rules:
        - Use the specific column names and table names from the schema.
        - Do not hallucinate columns.
        - If the goal requires multiple steps efficiently expressible in one query (e.g. CTEs), do so.
        - Return only the SELECT statement.
        - Respond with a valid JSON object containing:
          - "query": The SQL string.
          - "explanation": Brief explanation.
        """
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "Schema:\n{schema}\n\nGoal: {goal}\n\n{format_instructions}")
        ])
        
        chain = prompt | self.llm | parser
        try:
            result = chain.invoke({
                "schema": schema_context, 
                "goal": goal,
                "format_instructions": parser.get_format_instructions()
            })
            # Handle potential casing issues or extra keys
            return result.get('query') or result.get('sql') or ""
        except Exception as e:
            # Fallback text parsing if JSON fails (Gemini sometimes returns markdown json)
            print(f"SQL Gen Error: {e}")
            return ""
