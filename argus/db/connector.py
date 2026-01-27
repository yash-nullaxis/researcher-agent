from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import sqlalchemy
from sqlalchemy import create_engine, text, inspect
from contextlib import contextmanager

class DBConnector(ABC):
    @abstractmethod
    def get_schema_info(self) -> str:
        """Return schema information as a string."""
        pass
    
    @abstractmethod
    def execute_query(self, query: str) -> List[Dict[str, Any]]:
        """Execute a query and return results as a list of dicts."""
        pass

class SqlAlchemyConnector(DBConnector):
    def __init__(self, connection_str: str):
        self.engine = create_engine(connection_str)

    def get_schema_info(self) -> str:
        inspector = inspect(self.engine)
        schema_info = []
        for table_name in inspector.get_table_names():
            columns = inspector.get_columns(table_name)
            # Format: name (type)
            col_strs = [f"{c['name']} ({c['type']})" for c in columns]
            
            # Samples (Assuming generic SQL support)
            samples_str = ""
            row_count = "Unknown"
            try:
                with self.engine.connect() as conn:
                    # Row Count
                    count_res = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                    row_count = count_res.scalar()

                    # Samples
                    res = conn.execute(text(f"SELECT * FROM {table_name} LIMIT 3"))
                    samples = [dict(row) for row in res.mappings()]
                    samples_str = str(samples)
            except Exception as e:
                samples_str = f"No samples ({e})"
                if row_count == "Unknown":
                    row_count = f"Error: {e}"

            schema_info.append(f"Table: {table_name}\nRows: {row_count}\nColumns: {', '.join(col_strs)}\nSamples: {samples_str}")
        return "\n\n".join(schema_info)

    def execute_query(self, query: str) -> List[Dict[str, Any]]:
        from ..safety import SQLValidator
        is_valid, error = SQLValidator().validate(query)
        if not is_valid:
            raise ValueError(f"Safety Check Failed: {error}")

        with self.engine.connect() as conn:
            result = conn.execute(text(query))
            return [dict(row) for row in result.mappings()]

class DuckDBConnector(DBConnector):
    """
    Connector for local files (CSV, Parquet, JSON, Excel) using DuckDB.
    """
    def __init__(self, db_path: str = ":memory:", files: List[str] = None):
        import duckdb
        import pandas as pd
        self.conn = duckdb.connect(db_path)
        if files:
            for file_path in files:
                table_name = file_path.split("/")[-1].split(".")[0]
                if file_path.endswith('.xlsx'):
                    # Load Excel via pandas and register
                    df = pd.read_excel(file_path)
                    self.conn.register(table_name, df)
                else:
                    # CSV, Parquet, JSON
                    self.conn.execute(f"CREATE OR REPLACE VIEW {table_name} AS SELECT * FROM '{file_path}'")

    def get_schema_info(self) -> str:
        # Get all tables/views
        tables = self.conn.execute("SHOW TABLES").fetchall()
        table_names = [t[0] for t in tables]
        
        schema_info = []
        for table_name in table_names:
            columns = self.conn.execute(f"DESCRIBE {table_name}").fetchall()
            col_strs = [f"{c[0]} ({c[1]})" for c in columns]
            
            # Samples & Stats
            samples_str = ""
            row_count = "Unknown"
            try:
                # Row Count
                row_count = self.conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]

                # Samples
                sample_rows = self.conn.execute(f"SELECT * FROM {table_name} LIMIT 3").fetchall()
                sample_cols = [desc[0] for desc in self.conn.description]
                samples = [dict(zip(sample_cols, row)) for row in sample_rows]
                samples_str = str(samples)
            except Exception:
                samples_str = "No samples."

            schema_info.append(f"Table: {table_name}\nRows: {row_count}\nColumns: {', '.join(col_strs)}\nSamples: {samples_str}")
        
        return "\n\n".join(schema_info)

    def execute_query(self, query: str) -> List[Dict[str, Any]]:
        from ..safety import SQLValidator
        is_valid, error = SQLValidator().validate(query, dialect="duckdb")
        if not is_valid:
            raise ValueError(f"Safety Check Failed: {error}")

        # DuckDB returns list of tuples or we can convert to arrow/pandas/dict
        # Using fetchall() and description to make dicts
        cursor = self.conn.execute(query)
        result = cursor.fetchall()
        cols = [desc[0] for desc in cursor.description]
        return [dict(zip(cols, row)) for row in result]
    
    @property
    def engine(self):
        # DuckDB doesn't exactly have an sqlalchemy 'engine' object exposed this way easily 
        # unless we use duckdb_engine. 
        # For the SchemaInspector to work using `inspect(self.db.engine)`, we might need to adapt SchemaInspector
        # OR make this class strictly unrelated to SQLAlchemy inspection.
        # But SchemaInspector uses `inspect(self.db.engine)`. 
        # Refactor Plan: Update SchemaInspector to rely on `get_schema_info` or `get_columns` method of DBConnector
        # instead of direct sqlalchemy.inspect.
        return None
