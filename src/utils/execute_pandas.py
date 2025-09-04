# api/execute_pandas.py
import pandas as pd
import json
import logging

def execute_pandas_code(df: pd.DataFrame, operations: list) -> list:
    """
    Executes a list of pandas code snippets on a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to run the code on.
        operations (list): A list of dicts, each with 'id', 'title', 'pandas_code'.

    Returns:
        list: The list of operations with an added 'result' key for each.
    """
    results = []
    for op in operations:
        result_obj = op.copy()
        try:
            # Execute the pandas code. 'df' is available in the local scope.
            # Using eval is generally safer than exec for returning a value.
            result = eval(op['pandas_code'], {'pd': pd, 'df': df})

            # Convert result to a JSON-serializable format
            if isinstance(result, (pd.DataFrame, pd.Series)):
                # Using orient='records' for DataFrame is often useful for frontend
                result_json = json.loads(result.to_json(orient='split' if isinstance(result, pd.DataFrame) else 'records'))
            else:
                result_json = result

            result_obj['result'] = result_json
        except Exception as e:
            logging.error(f"Error executing pandas code for id {op.get('id')}: {op.get('pandas_code')}. Error: {e}")
            result_obj['result'] = {"error": str(e)}
        results.append(result_obj)
    return results