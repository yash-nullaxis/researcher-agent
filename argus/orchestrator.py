import logging
from typing import Dict, Optional

from langgraph.graph import END, StateGraph
from langgraph.types import Send

from .analyst.decomposer import QueryDecomposer
from .analyst.executor import SQLExecutor
from .analyst.relevance import RelevanceChecker
from .analyst.schema import SchemaInspector
from .analyst.sql_gen import SQLSynthesizer
from .config import AgentConfig
from .db.connector import DBConnector, DuckDBConnector, SqlAlchemyConnector
from .safety import SQLValidator
from .state import AnalysisState, AnalysisStep, StepResult, AnalystInput
from .synthesizer.memo import Synthesizer


logger = logging.getLogger(__name__)


class Orchestrator:
    def __init__(
        self,
        config: AgentConfig,
        connector: Optional[DBConnector] = None,
        connectors: Optional[Dict[str, DBConnector]] = None,
    ):
        self.config = config
        
        # Initialize Services
        if connectors:
            self.connectors = connectors
            self.default_source = list(connectors.keys())[0]
            # Inspector handles dict
            # Check for a specific 'rag_store' or similar in connectors? 
            # Or dedicated arg? For now, we assume no RAG unless explicitly wired not seeing it in init.
            # Let's assume one key matches 'vector_store' convention or we add explicit arg.
            # Adding explicit arg `vector_store_connector` to init would be cleaner but changing sig again.
            # Let's check if any connector is ChromaDB type?
            # Creating RAG hook:
            rag_store = connectors.get('metadata_store') # Convention
            self.inspector = SchemaInspector(self.connectors, vector_store=rag_store) 
            # Executor needs to know which one to use at runtime, so we pass the registry or handle in node
            self.executor = SQLExecutor(self.connectors[self.default_source]) # Default for now
        elif connector:
            self.connectors = {"default": connector}
            self.default_source = "default"
            self.inspector = SchemaInspector(connector)
            self.executor = SQLExecutor(connector)
        else:
            default_conn = SqlAlchemyConnector(config.db_connection_str)
            self.connectors = {"default": default_conn}
            self.default_source = "default"
            self.inspector = SchemaInspector(default_conn)
            self.executor = SQLExecutor(default_conn)

        self.decomposer = QueryDecomposer(config)
        self.sql_gen = SQLSynthesizer(config)
        self.validator = SQLValidator()
        # self.executor is already initialized in if/else block above
        self.relevance_checker = RelevanceChecker(config)
        self.synthesizer = Synthesizer(config)
        
        # Compile Graph
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(AnalysisState)
        
        workflow.add_node("planner", self.planner_node)
        workflow.add_node("analyst", self.analyst_node)
        workflow.add_node("synthesizer", self.synthesizer_node)
        
        workflow.set_entry_point("planner")
        
        # Planner decides what to do next (start analysis)
        workflow.add_conditional_edges("planner", self.scheduler)
        
        # After each analyst step finishes, we check if more steps are unblocked
        workflow.add_conditional_edges("analyst", self.scheduler)
        
        workflow.add_edge("synthesizer", END)
        
        return workflow.compile()

    def _infer_sqlglot_dialect(self, connector: DBConnector) -> str:
        """
        Infer a sqlglot-compatible dialect name for the active connector.

        This is used both to:
        - steer SQL generation (LLM prompt)
        - validate/parse SQL locally before execution
        """
        # DuckDB connector is explicit
        if isinstance(connector, DuckDBConnector):
            return "duckdb"

        # SQLAlchemy engines expose a dialect name (e.g. 'postgresql', 'duckdb', 'sqlite')
        engine = getattr(connector, "engine", None)
        dialect_name = None
        try:
            if engine is not None and hasattr(engine, "dialect"):
                dialect_name = getattr(engine.dialect, "name", None)
        except Exception:
            dialect_name = None

        if not dialect_name:
            return "duckdb"

        # Map SQLAlchemy dialect names to sqlglot names
        if dialect_name in ("postgresql", "postgres"):
            return "postgres"
        if dialect_name == "duckdb":
            return "duckdb"
        if dialect_name in ("sqlite",):
            return "sqlite"

        # Best-effort fallback
        return str(dialect_name)

    def planner_node(self, state: AnalysisState):
        if self.config.verbose:
            logger.info("[Planner] Fetching Schema Summary...")
            
        schema_summary = self.inspector.get_summary(query=state["user_query"])
        # Structured schema for validators / planner hints
        schema_dict = {}
        try:
            # Single connector vs multi‑source dict
            if isinstance(self.connectors, dict):
                for name, conn in self.connectors.items():
                    try:
                        schema_dict[name] = conn.get_schema_dict()
                    except Exception:
                        schema_dict[name] = {}
            else:
                schema_dict = {"default": self.connectors.get_schema_dict()}  # type: ignore[union-attr]
        except Exception:
            schema_dict = {}
        
        if self.config.verbose:
            logger.info(f"[Planner] Schema Summary:\n{schema_summary}")
        
        # Decompose
        datasource_list = list(self.connectors.keys())
        steps = self.decomposer.decompose(
            state["user_query"],
            context=schema_summary,
            datasources=self.config.datasource_hints or {
                name: "" for name in datasource_list
            },
        )

        if not steps:
            # Fallback to a single default SQL step so the agent remains usable.
            steps = [
                AnalysisStep(
                    id=1,
                    description=state["user_query"],
                    tool="sql",
                    datasource=self.default_source,
                    dependency=-1,
                    thought="Fallback single-step plan because planner returned no steps.",
                )
            ]
        
        if self.config.verbose:
            logger.info(f"[Planner] Generated Plan: {len(steps)} steps")
            for s in steps:
                logger.info(f"  - [{s.id}] {s.description} (Source: {s.datasource})")
                
        debug_msgs = [
            "[Planner] Completed planning.",
            f"[Planner] Sources: {list(self.connectors.keys())}",
        ]

        # Reset any results/state for a fresh run (though usually this filters through init state)
        return {
            "plan": steps,
            "context_data": {"schema": schema_summary, "schema_dict": schema_dict},
            "debug_trace": debug_msgs,
        }

    def scheduler(self, state: AnalysisState):
        """
        Determines which steps are ready to run in parallel.
        """
        plan = state.get("plan", [])
        results = state.get("step_results", [])
        
        # IDs of steps that have successfully finished
        completed_ids = {r.step_id for r in results if r.success}
        # IDs of steps that failed (we won't retry them infinitely here, we just block dependents)
        failed_ids = {r.step_id for r in results if not r.success}
        
        attempted_ids = completed_ids.union(failed_ids)
        
        tasks = []
        for step in plan:
            # If already attempted, skip
            if step.id in attempted_ids:
                continue
            
            # Check dependencies
            # Run if:
            # 1. No dependency (-1)
            # 2. Dependency is in completed_ids (successfully finished)
            if step.dependency == -1 or step.dependency in completed_ids:
                tasks.append(Send("analyst", {
                    "step": step, 
                    "user_query": state["user_query"], 
                    "context_data": state.get("context_data", {})
                }))
            elif step.dependency in failed_ids:
                # Dependency failed, so this step cannot run. 
                # Ideally we should mark this step as skipped/failed too, but for now we just don't schedule it.
                # It will eventually result in no tasks -> synthesizer.
                pass
                
        if tasks:
            return tasks
            
        return "synthesizer"

    def analyst_node(self, input_data: AnalystInput):
        """
        Execute a single analysis step. 
        Note: input_data is NOT the global AnalysisState, but a specific payload from Send().
        """
        step = input_data["step"]
        user_query = input_data["user_query"]
        context_data = input_data["context_data"]
        
        target_source = step.datasource or self.default_source
        if target_source not in self.connectors:
             target_source = self.default_source
             
        self.executor.db = self.connectors[target_source]
        
        debug_msgs = []
        
        if step.tool == "sql":
            schema_context = context_data.get("schema", "")
            
            # Local Retry Loop for Robustness
            last_error = None
            for attempt in range(self.config.max_retry_per_step):
                try:
                    if self.config.verbose:
                         logger.info(f"[Analyst] Step {step.id} (Attempt {attempt + 1})")

                    # Generate SQL
                    dialect = self._infer_sqlglot_dialect(self.connectors[target_source])
                    query = self.sql_gen.generate_sql(
                        step.description, 
                        schema_context,
                        dialect=dialect,
                        error_context=last_error,
                    )
                    
                    debug_msgs.append(f"[Analyst] Step {step.id} SQL: {query}")

                    # Validate
                    connector = self.connectors[target_source]
                    try:
                        connector_schema = (
                            connector.get_schema_dict()
                            if hasattr(connector, "get_schema_dict")
                            else {}
                        )
                    except Exception:
                        connector_schema = {}
                        
                    is_valid, error_msg = self.validator.validate(
                        query, dialect=dialect, schema_info=connector_schema
                    )
                    
                    if not is_valid:
                        last_error = f"Validation Error: {error_msg}"
                        debug_msgs.append(f"[Analyst] Validation failed: {error_msg}")
                        continue # Retry

                    # Execute
                    result = self.executor.execute(query)
                    
                    # Relevance Check
                    rel = self.relevance_checker.check_relevance(
                        query, step.description, str(result)[:1000]
                    )
                    
                    if not rel.is_relevant or rel.score < self.config.min_relevance_score:
                        last_error = f"Low relevance: {rel.reasoning}"
                        debug_msgs.append(f"[Analyst] Low relevance: {last_error}")
                        continue # Retry
                        
                    # Success
                    row_count = len(result) if isinstance(result, list) else None
                    columns = list(result[0].keys()) if row_count and row_count > 0 else None
                    
                    return {
                        "step_results": [StepResult(
                            step_id=step.id,
                            output=result,
                            success=True,
                            query_executed=query,
                            row_count=row_count,
                            columns=columns
                        )],
                        "debug_trace": debug_msgs
                    }

                except Exception as e:
                    last_error = str(e)
                    debug_msgs.append(f"[Analyst] Error attempt {attempt+1}: {e}")
            
            # specific failure after retries
            return {
                "step_results": [StepResult(
                    step_id=step.id,
                    output=None,
                    success=False,
                    error=last_error or "Max retries exceeded"
                )],
                 "debug_trace": debug_msgs
            }
        else:
            return {
                "step_results": [StepResult(
                    step_id=step.id,
                    output="Skipped non-SQL step",
                    success=False,
                    error="Unsupported tool"
                )],
                "debug_trace": [f"[Analyst] Skipped non-SQL step {step.id}"]
            }

    async def synthesizer_node(self, state: AnalysisState):
        memo = await self.synthesizer.synthesize(state)
        return {"final_memo": memo}
        
    async def run(self, query: str):
        """
        Main entry point to run the agent.
        """
        initial_state: AnalysisState = {
            "user_query": query,
            "step_results": [],
            "current_step_index": 0, # Kept for structure but unused
            "context_data": {},
            "debug_trace": ["[Run] Starting analysis."],
        }
        return await self.graph.ainvoke(initial_state)

    async def stream_run(self, query: str):
        """
        Runs the agent and streams the final memo generation.
        """
        initial_state: AnalysisState = {
            "user_query": query,
            "step_results": [],
            "current_step_index": 0,
            "context_data": {},
            "debug_trace": ["[Run] Starting analysis."],
        }
        
        async for event in self.graph.astream_events(initial_state, version="v1"):
            kind = event["event"]
            
            # Stream tokens from the synthesizer node
            if kind == "on_chat_model_stream":
                metadata = event.get("metadata", {})
                # We want to catch the LLM stream inside the 'synthesizer' node
                if metadata.get("langgraph_node") == "synthesizer":
                    content = event["data"]["chunk"].content
                    if content:
                        yield content
