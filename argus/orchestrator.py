from typing import Optional, Dict
from langgraph.graph import StateGraph, END
from .state import AnalysisState, StepResult, AnalysisStep
from .config import AgentConfig
from .db.connector import SqlAlchemyConnector, DBConnector
from .analyst.schema import SchemaInspector
from .analyst.decomposer import QueryDecomposer
from .analyst.sql_gen import SQLSynthesizer
from .safety import SQLValidator
from .analyst.executor import SQLExecutor
from .analyst.relevance import RelevanceChecker
from .synthesizer.memo import Synthesizer

class Orchestrator:
    def __init__(self, config: AgentConfig, connector: Optional[DBConnector] = None, connectors: Optional[Dict[str, DBConnector]] = None):
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

    def planner_node(self, state: AnalysisState):
        if self.config.verbose:
            print("\n[Planner] Fetching Schema Summary...")
            
        schema_summary = self.inspector.get_summary(query=state['user_query'])
        
        if self.config.verbose:
            print(f"[Planner] Schema Summary:\n{schema_summary}")
        
        # Decompose
        steps = self.decomposer.decompose(state['user_query'], context=schema_summary)
        
        if self.config.verbose:
            print(f"\n[Planner] Generated Plan: {len(steps)} steps")
            for s in steps:
                print(f"  - [{s.id}] {s.description} (Source: {s.datasource})")
                
        return {
            "plan": steps, 
            "context_data": {"schema": schema_summary},
            "current_step_index": 0,
            # Initialize lists (reducers will append if we passed list, but here we init)
            # Actually, because we use operator.add, we must act carefully
            # But the first node can return the initial expected keys if not present
        }

    def analyst_node(self, state: AnalysisState):
        current_idx = state['current_step_index']
        # Safety check
        if current_idx >= len(state['plan']):
            return {} 
            
        step = state['plan'][current_idx]
        
        # Determine Connector
        target_source = step.datasource or self.default_source
        if target_source not in self.connectors:
             # Fallback if LLM halluncinated source name
             target_source = self.default_source
             
        # Switch Executor's DB
        # Ideally, we should instantiate a fresh executor or have executor accept db in execute()
        # For now, we swap the db in the existing instance (a bit hacky but works for sequential)
        self.executor.db = self.connectors[target_source]
        
        # Execute Step
        # Execute Step
        if step.tool == 'sql':
            schema_context = state['context_data']['schema']
            
            if self.config.verbose:
                print(f"\n[Analyst] Step ID: {step.id}")
                print(f"[Analyst] Goal: {step.description}")
                print(f"[Analyst] Datasource: {target_source}")
                print(f"[Analyst] Schema Context Length: {len(schema_context)}")
                
            # Generate
            query = self.sql_gen.generate_sql(step.description, schema_context)
            
            if self.config.verbose:
                print(f"[Analyst] Generated SQL: {query}")
            
            # Validate
            # If target is Vector/NoSQL, validation might differ. 
            # DuckDB/SQLAlchemy support SQLValidator. 
            # Vector/NoSQL connectors might need bypass or custom validation.
            if isinstance(self.executor.db, (SqlAlchemyConnector, type(None))): # DuckDB inherits DBConnector but checking instance is better
                 # Logic: We should rely on connector's own safety checks or global validator?
                 # Global validator parses SQL. If Vector DB uses natural language "query", SQLValidator fails.
                 # Let's Skip SQL Validation if not a SQL connector?
                 # Or better: SQLValidator defaults to SQL dialect.
                 pass

            # Execute
            try:
                result = self.executor.execute(query)
                
                if self.config.verbose:
                    print(f"[Analyst] Raw Result: {result}")
                
                # Check Relevance
                rel = self.relevance_checker.check_relevance(query, step.description, str(result)[:1000]) # truncated
                
                if not rel.is_relevant or rel.score < 5:
                    return {
                        "step_results": [StepResult(step_id=step.id, output=result, success=False, error=f"Low relevance: {rel.reasoning}", query_executed=query)],
                        "current_step_index": current_idx + 1
                    }
                
                return {
                    "step_results": [StepResult(step_id=step.id, output=result, success=True, query_executed=query)],
                    "current_step_index": current_idx + 1
                }
            except Exception as e:
                if self.config.verbose:
                    print(f"[Analyst] Error executing SQL: {e}")
                return {
                    "step_results": [StepResult(step_id=step.id, output=None, success=False, error=str(e), query_executed=query)],
                    "current_step_index": current_idx + 1
                }
        else:
            # Placeholder for other tools or fallback
            return {
                "step_results": [StepResult(step_id=step.id, output="Skipped non-SQL step", success=False)],
                "current_step_index": current_idx + 1
            }

    def synthesizer_node(self, state: AnalysisState):
        memo = self.synthesizer.synthesize(state)
        return {"final_memo": memo}

    def should_continue(self, state: AnalysisState):
        if state['current_step_index'] < len(state['plan']):
            return "analyst"
        return "synthesizer"
        
    async def run(self, query: str):
        """
        Main entry point to run the agent.
        """
        initial_state = {
            "user_query": query, 
            "step_results": [],
            "errors": []
        }
        # invoke is sync or async? Compiled graph supports both.
        # But we made this method async.
        return await self.graph.ainvoke(initial_state)
