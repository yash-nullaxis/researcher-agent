from typing import Dict, Optional

from langgraph.graph import END, StateGraph

from .analyst.classifier import QueryClassifier
from .analyst.decomposer import QueryDecomposer
from .analyst.executor import SQLExecutor
from .analyst.relevance import RelevanceChecker
from .analyst.schema import SchemaInspector
from .analyst.sql_gen import SQLSynthesizer
from .config import AgentConfig
from .db.connector import DBConnector, DuckDBConnector, SqlAlchemyConnector
from .safety import SQLValidator
from .state import AnalysisState, AnalysisStep, StepResult
from .synthesizer.memo import Synthesizer


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
        self.classifier = QueryClassifier(config)
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
        
        workflow.add_edge("planner", "analyst")
        
        workflow.add_conditional_edges(
            "analyst",
            self.should_continue,
            {
                "analyst": "analyst",
                "synthesizer": "synthesizer"
            }
        )
        
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
            print("\n[Planner] Fetching Schema Summary...")
            
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
            print(f"[Planner] Schema Summary:\n{schema_summary}")
        
        # Decompose
        datasource_list = list(self.connectors.keys())
        
        # Classify query complexity
        if self.config.verbose:
            print(f"[Planner] Classifying query complexity...")
            
        complexity = self.classifier.classify(state["user_query"])
        
        if not complexity.is_complex:
            if self.config.verbose:
                print(f"[Planner] Query classified as SIMPLE. Reason: {complexity.reasoning}")
                print(f"[Planner] Skipping decomposition, creating direct SQL step.")
            
            steps = [
                AnalysisStep(
                    id=1,
                    description=state["user_query"],
                    tool="sql",
                    datasource=self.default_source, # Use default source for simple queries
                    dependency=-1,
                    thought=f"Simple query detected ({complexity.reasoning}). Executing directly.",
                )
            ]
        else:
            if self.config.verbose:
                print(f"[Planner] Query classified as COMPLEX. Reason: {complexity.reasoning}")
                print(f"[Planner] Proceeding with decomposition.")
                
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
            print(f"\n[Planner] Generated Plan: {len(steps)} steps")
            for s in steps:
                print(f"  - [{s.id}] {s.description} (Source: {s.datasource})")
                
        debug_msgs = [
            "[Planner] Completed planning.",
            f"[Planner] Sources: {list(self.connectors.keys())}",
        ]

        return {
            "plan": steps,
            "context_data": {"schema": schema_summary, "schema_dict": schema_dict},
            "current_step_index": 0,
            "debug_trace": debug_msgs,
        }

    def analyst_node(self, state: AnalysisState):
        current_idx = state["current_step_index"]
        if current_idx >= len(state["plan"]):
            return {} 
            
        step = state["plan"][current_idx]
        target_source = step.datasource or self.default_source
        if target_source not in self.connectors:
             target_source = self.default_source
             
        self.executor.db = self.connectors[target_source]
        
        if step.tool == "sql":
            schema_context = state["context_data"].get("schema", "")
            schema_dict = state["context_data"].get("schema_dict", {})

            # Check if we've already failed this step too many times
            if state.get("last_error") and state.get(
                "retry_count", 0
            ) >= self.config.max_retry_per_step:
                if self.config.verbose:
                    print(f"[Analyst] Max retries reached for step {step.id}")
                return {
                    "step_results": [
                        StepResult(
                            step_id=step.id,
                            success=False,
                            error=state["last_error"],
                            output=None,
                        )
                    ],
                    "current_step_index": current_idx + 1,
                    "last_error": None,
                    "retry_count": 0,
                    "debug_trace": [
                        f"[Analyst] Max retries reached for step {step.id}: {state.get('last_error')}"
                    ],
                }

            if self.config.verbose:
                print(
                    f"\n[Analyst] Step ID: {step.id} (Attempt {state.get('retry_count', 0) + 1})"  # type: ignore[arg-type]
                )
                
            # Generate with Error Context if present
            dialect = self._infer_sqlglot_dialect(self.connectors[target_source])
            query = self.sql_gen.generate_sql(
                step.description, 
                schema_context,
                dialect=dialect,
                error_context=state.get("last_error"),
            )
            
            if self.config.verbose:
                print(f"[Analyst] Generated SQL: {query}")

            debug_msgs = [f"[Analyst] Step {step.id} SQL: {query}"]

            # Fast local validation before hitting the DB
            connector = self.connectors[target_source]
            try:
                connector_schema = (
                    connector.get_schema_dict()
                    if hasattr(connector, "get_schema_dict")
                    else {}
                )
            except Exception:
                connector_schema = {}

            # Choose dialect based on connector type/engine
            dialect = self._infer_sqlglot_dialect(connector)
            is_valid, error_msg = self.validator.validate(
                query, dialect=dialect, schema_info=connector_schema
            )
            if not is_valid:
                if self.config.verbose:
                    print(f"[Analyst] Local validation failed: {error_msg}")
                debug_msgs.append(
                    f"[Analyst] Validation failed for step {step.id}: {error_msg}"
                )
                return {
                    "last_error": error_msg,
                    "retry_count": state.get("retry_count", 0) + 1,
                    "debug_trace": debug_msgs,
                }
            
            try:
                result = self.executor.execute(query)
                
                # Check Relevance
                rel = self.relevance_checker.check_relevance(
                    query, step.description, str(result)[:1000]
                )
                
                if not rel.is_relevant or rel.score < self.config.min_relevance_score:
                    error_msg = f"Low relevance: {rel.reasoning}"
                    debug_msgs.append(
                        f"[Analyst] Low relevance for step {step.id}: {error_msg}"
                    )
                    return {
                        "last_error": error_msg,
                        "retry_count": state.get("retry_count", 0) + 1,
                        "debug_trace": debug_msgs,
                    }

                # Basic per-step metrics
                row_count = len(result) if isinstance(result, list) else None
                columns = list(result[0].keys()) if row_count and row_count > 0 else None

                # Success
                return {
                    "step_results": [
                        StepResult(
                            step_id=step.id,
                            output=result,
                            success=True,
                            query_executed=query,
                            row_count=row_count,
                            columns=columns,
                        )
                    ],
                    "current_step_index": current_idx + 1,
                    "last_error": None,
                    "retry_count": 0,
                    "total_iterations": state.get("total_iterations", 0) + 1,
                    "debug_trace": debug_msgs,
                }
            except Exception as e:
                if self.config.verbose:
                    print(f"[Analyst] Execution Error: {e}")
                return {
                    "last_error": str(e),
                    "retry_count": state.get("retry_count", 0) + 1,
                    "total_iterations": state.get("total_iterations", 0) + 1,
                    "debug_trace": [
                        f"[Analyst] Execution error for step {step.id}: {e}"
                    ],
                }
        else:
            return {
                "step_results": [
                    StepResult(
                        step_id=step.id,
                        output="Skipped non-SQL step",
                        success=False,
                    )
                ],
                "current_step_index": current_idx + 1,
                "last_error": None,
                "retry_count": 0,
                "total_iterations": state.get("total_iterations", 0) + 1,
                "debug_trace": [
                    f"[Analyst] Skipped non-SQL step {step.id} with tool '{step.tool}'"
                ],
            }

    def synthesizer_node(self, state: AnalysisState):
        memo = self.synthesizer.synthesize(state)
        return {"final_memo": memo}

    def should_continue(self, state: AnalysisState):
        """
        Decide whether to keep running the analyst node or hand off to the synthesizer.

        Important invariants:
        - If we've exhausted the plan or hit max_steps, we must terminate, even if
          there is a lingering last_error. This prevents infinite no-op loops.
        - Auto-repair (retrying analyst) is only allowed while there are remaining
          steps and we haven't exceeded the recursion_limit.
        """

        current_idx = state.get("current_step_index", 0)
        plan_len = len(state.get("plan", []) or [])

        # 1) Hard stops: plan exhausted or global max_steps reached.
        if current_idx >= plan_len or current_idx >= self.config.max_steps:
            return "synthesizer"

        # 2) Auto-repair cycle while there are still steps left.
        if state.get("last_error"):
            if state.get("total_iterations", 0) >= self.config.recursion_limit:
                # Stop auto-repair if we've exceeded recursion limit.
                return "synthesizer"
            return "analyst"

        # 3) Normal forward progress through remaining steps.
        return "analyst"
        
    async def run(self, query: str):
        """
        Main entry point to run the agent.
        """
        initial_state: AnalysisState = {
            "user_query": query,
            "step_results": [],
            "errors": [],
            "current_step_index": 0,
            "retry_count": 0,
            "last_error": None,
            "context_data": {},
            "total_iterations": 0,
            "debug_trace": ["[Run] Starting analysis."],
        }
        return await self.graph.ainvoke(initial_state)
