import sqlglot
from sqlglot import exp

class SQLValidator:
    def validate(self, sql: str, dialect: str = "duckdb") -> tuple[bool, str]:
        """
        Validates SQL syntax and ensures strict read-only safety.
        Returns: (is_valid, error_message)
        """
        try:
            # Parse
            parsed = sqlglot.parse_one(sql, read=dialect)
        except Exception as e:
            return False, f"Syntax Error: {str(e)}"
            
        # Safety Check: Block mutations
        # We explicitly check for modification statements
        # Using a list of classes that exist in sqlglot.exp
        forbidden_types = []
        for type_name in ["Drop", "Delete", "Update", "Insert", "Create", "Merge", "Commit", "Rollback", "AlterTable", "AlterColumn", "TruncateTable"]:
            if hasattr(exp, type_name):
                forbidden_types.append(getattr(exp, type_name))
        
        # Grant/Revoke are trickier in some versions. 
        # Let's also block by string if they are not in exp.
        if parsed.find(tuple(forbidden_types)):
            return False, "Security Violation: Query contains forbidden mutation or DDL statements."
            
        # Fallback string check for common DDL/DML not caught by parser classes
        sql_upper = sql.upper()
        for cmd in ["GRANT ", "REVOKE ", "DROP ", "DELETE ", "UPDATE ", "INSERT ", "CREATE ", "ALTER ", "TRUNCATE ", "MERGE "]:
            if sql_upper.startswith(cmd) or f"\n{cmd}" in sql_upper or f";{cmd}" in sql_upper:
                 return False, f"Security Violation: Query attempts {cmd.strip()} operation."

            
        return True, ""
