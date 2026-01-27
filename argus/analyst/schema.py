from ..db.connector import DBConnector
from typing import Dict, Optional
import uuid

class SchemaInspector:
    def __init__(self, db: DBConnector | Dict[str, DBConnector], vector_store: Optional[DBConnector] = None):
        self.db = db
        self.vector_store = vector_store
        
        # Trigger indexing if RAG is enabled
        if self.vector_store:
             self._index_schemas()

    def get_summary(self, query: str = None) -> str:
        """
        Returns a natural language summary of the database schema with profiling stats.
        If query is provided and RAG is enabled, returns relevant tables only to save context.
        """
        full_summary = self._get_global_summary()

        if query and self.vector_store:
            # Perform RAG retrieval
            try:
                # We assume execute_query on Chroma connector returns relevant doc content
                # ChromaDBConnector returns list of dicts with 'content' and 'metadata'
                results = self.vector_store.execute_query(query)
                if results:
                     context_parts = []
                     for r in results:
                         content = r.get('content', '')
                         meta = r.get('metadata', {})
                         context_parts.append(f"{content}\n(Metadata: {meta})")
                     
                     if context_parts:
                        return "--- RELEVANT TABLES (RAG FILTERED) ---\n" + "\n\n".join(context_parts)
            except Exception as e:
                 # Fallback to full summary on error
                 return f"{full_summary}\n(RAG Error: {e})"
        
        return full_summary

    def _index_schemas(self):
        """
        Indexes table schemas into the vector store for RAG.
        This parses the text summary and chunks it by table (heuristic).
        """
        try:
            full_text = self._get_global_summary()
            
            # Simple Heuristic Chunking: Split by "Table:" or "Collection:"
            # A better way would be for Connectors to return structured metadata, but we work with text for now.
            chunks = []
            metadatas = []
            ids = []
            
            # Split by double newline which usually separates tables in our format
            raw_sections = full_text.split("\n\n")
            
            for section in raw_sections:
                if not section.strip(): continue
                
                # Metadata extraction
                source = "unknown"
                if "--- SOURCE:" in section:
                    try:
                        source = section.split("SOURCE:")[1].split("---")[0].strip()
                    except: pass
                
                chunks.append(section)
                metadatas.append({"source": source, "type": "schema_summary"})
                ids.append(str(uuid.uuid4()))
            
            # Upsert to Vector Store
            if hasattr(self.vector_store, 'add_documents'):
                self.vector_store.add_documents(documents=chunks, metadatas=metadatas, ids=ids)
                
        except Exception as e:
            print(f"Warning: Failed to index schemas for RAG: {e}")

    def _get_global_summary(self) -> str:
        """
        Aggregates schema info from all connectors.
        """
        if isinstance(self.db, dict):
            # Multi-source mode
            global_summary = []
            for name, connector in self.db.items():
                try:
                    schema = connector.get_schema_info()
                    global_summary.append(f"--- SOURCE: {name} ---\n{schema}")
                except Exception as e:
                    global_summary.append(f"--- SOURCE: {name} ---\nError fetching schema: {e}")
            return "\n\n".join(global_summary)
        else:
            return self.db.get_schema_info()
