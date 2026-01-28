import operator
from typing import TypedDict, List, Dict, Any, Optional, Annotated
from pydantic import BaseModel, Field

class AnalysisStep(BaseModel):
    id: int = Field(description="Unique identifier for the step")
    description: str = Field(description="Detailed description of the analysis step")
    tool: str = Field(description="Tool to use, e.g. 'sql'")
    datasource: str = Field(default="default", description="Data source name to target")
    dependency: int = Field(default=-1, description="ID of a previous step this depends on, or -1 if none")
    thought: str = Field(default="", description="Reasoning for this step")

class StepResult(BaseModel):
    step_id: int
    output: Any
    success: bool
    error: Optional[str] = None
    query_executed: Optional[str] = None

class AnalysisState(TypedDict):
    """
    Global state for the Argus Analysis Agent.
    """
    user_query: str
    plan: List[AnalysisStep]
    current_step_index: int
    context_data: Dict[str, Any]  # Stores schema info, sample data, etc.
    step_results: Annotated[List[StepResult], operator.add]
    final_memo: str
    errors: Annotated[List[str], operator.add]
    retry_count: int  # Tracking attempts for current step
    last_error: Optional[str]  # Feedback for error correction
