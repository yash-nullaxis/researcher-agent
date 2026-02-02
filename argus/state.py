import operator
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from pydantic import BaseModel, Field


class AnalysisStep(BaseModel):
    id: int = Field(description="Unique identifier for the step")
    description: str = Field(description="Detailed description of the analysis step")
    tool: str = Field(description="Tool to use, e.g. 'sql'")
    datasource: str = Field(
        default="default", description="Data source name to target"
    )
    dependency: int = Field(
        default=-1, description="ID of a previous step this depends on, or -1 if none"
    )
    thought: str = Field(default="", description="Reasoning for this step")


class StepResult(BaseModel):
    step_id: int
    output: Any
    success: bool
    error: Optional[str] = None
    query_executed: Optional[str] = None
    row_count: Optional[int] = None
    columns: Optional[List[str]] = None


class AnalysisState(TypedDict, total=False):
    """
    Global state for the Argus Analysis Agent.
    """

    # Core
    user_query: str
    plan: List[AnalysisStep]
    context_data: Dict[str, Any]  # Stores schema info, sample data, etc.

    # Outputs / accumulation
    step_results: Annotated[List[StepResult], operator.add]
    final_memo: str
    
    # Debug / tracing
    debug_trace: Annotated[List[str], operator.add]

# Input schema for the Analyst Node (for parallel execution)
class AnalystInput(TypedDict):
    step: AnalysisStep
    user_query: str
    context_data: Dict[str, Any]
