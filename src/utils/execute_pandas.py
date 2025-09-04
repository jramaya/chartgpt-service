# api/execute_pandas.py
import pandas as pd
import numpy as np
import json
import logging

def execute_pandas_code(df: pd.DataFrame, operation: dict) -> dict:
    """
    Executes a single pandas code snippet on a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to run the code on.
        operation (dict): A dict with 'id', 'title', 'pandas_code'.

    Returns:
        dict: The operation with an added 'result' key.
    
    Raises:
        Exception: If the pandas code execution fails.
    """
    result_obj = operation.copy()
    try:
        result = eval(operation['pandas_code'], {'pd': pd, 'df': df, 'np': np})

        if isinstance(result, (pd.DataFrame, pd.Series)):
            result_json = json.loads(result.to_json(orient='split' if isinstance(result, pd.DataFrame) else 'records'))
        else:
            result_json = result

        result_obj['result'] = result_json
        return result_obj
    except Exception as e:
        logging.error(f"Error executing pandas code for id {operation.get('id')}: {operation.get('pandas_code')}. Error: {e}")
        raise  # Re-raise the exception to be handled by the caller