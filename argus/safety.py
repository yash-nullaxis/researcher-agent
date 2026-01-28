import sqlglot
from sqlglot import exp

class SQLValidator:
    def validate(self, sql: str, dialect: str = "duckdb", schema_info: dict[str, list[str]] = None) -> tuple[bool, str]:
        """
        Validates SQL syntax, strict read-only safety, and optionally schema existence.
        schema_info: { table_name: [col1, col2, ...] }
        Returns: (is_valid, error_message)
        """
        try:
            # Parse
            parsed = sqlglot.parse_one(sql, read=dialect)
        except Exception as e:
            return False, f"Syntax Error: {str(e)}"
            
        # 1. Safety Check: Block mutations
        forbidden_types = []
        for type_name in ["Drop", "Delete", "Update", "Insert", "Create", "Merge", "Commit", "Rollback", "AlterTable", "AlterColumn", "TruncateTable"]:
            if hasattr(exp, type_name):
                forbidden_types.append(getattr(exp, type_name))
        
        if parsed.find(tuple(forbidden_types)):
            return False, "Security Violation: Query contains forbidden mutation or DDL statements."
            
        # Fallback string check
        sql_upper = sql.upper()
        for cmd in ["GRANT ", "REVOKE ", "DROP ", "DELETE ", "UPDATE ", "INSERT ", "CREATE ", "ALTER ", "TRUNCATE ", "MERGE "]:
            if sql_upper.startswith(cmd) or f"\n{cmd}" in sql_upper or f";{cmd}" in sql_upper:
                 return False, f"Security Violation: Query attempts {cmd.strip()} operation."

        # 2. Schema Verification (if provided)
        if schema_info:
            # Normalize schema keys to lowercase for robust matching
            schema_lower = {k.lower(): [c.lower() for c in v] for k, v in schema_info.items()}
            
            # Find all CTE names to ignore them as "tables"
            ctes = []
            for cte in parsed.find_all(exp.CTE):
                ctes.append(cte.alias.lower())
            
            # Check Tables
            query_tables = []
            for table in parsed.find_all(exp.Table):
                t_name = table.name.lower()
                if t_name not in ctes:
                    if t_name not in schema_lower:
                        return False, f"Schema Error: Table '{t_name}' does not exist."
                    query_tables.append(t_name)

            # Check Columns (best effort)
            for column in parsed.find_all(exp.Column):
                col_name = column.name.lower()
                col_table = column.table.lower()
                
                if col_table:
                    # If aliased table, we'd need to resolve alias. 
                    # For now, if col_table is in schema or is a query table, check it.
                    if col_table in schema_lower:
                         if col_name not in schema_lower[col_table]:
                             return False, f"Schema Error: Column '{col_name}' does not exist in table '{col_table}'."
                else:
                    # Unqualified column. Check if it exists in ANY table involved in the query.
                    # This handles simple queries well.
                    exists = False
                    for t in query_tables:
                        if col_name in schema_lower.get(t, []):
                            exists = True
                            break
                    
                    # If not in query tables, maybe it's a CTE column or something else? 
                    # We only error if we are sure it's not in any known physical table.
                    if query_tables and not exists:
                         return False, f"Schema Error: Column '{col_name}' not found in tables {query_tables}."

        return True, ""
