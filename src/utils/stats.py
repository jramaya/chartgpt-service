# src/utils/stats.py
import pandas as pd
import numpy as np
import json
import logging
from typing import Dict, Any, List

def generate_info(df: pd.DataFrame) -> Dict[str, Any]:
    """Generates detailed information about the DataFrame, similar to df.info()."""
    info = {}
    try:
        if df.empty:
            return {"message": "DataFrame is empty."}

        # Basic info
        info["class"] = df.__class__.__name__
        info["shape"] = list(df.shape)
        
        # Index info
        info["index"] = {
            "type": str(df.index.__class__.__name__),
            "start": df.index[0],
            "stop": df.index[-1] + 1 if isinstance(df.index, pd.RangeIndex) else None,
            "step": df.index.step if hasattr(df.index, 'step') else 1
        }
        
        # Columns info
        column_info = []
        for col in df.columns:
            column_info.append({
                "name": col,
                "dtype": str(df[col].dtype),
                "non_null": int(df[col].count()),
                "null": int(df[col].isnull().sum()),
                "unique": df[col].nunique(),
                "memory_usage_bytes": int(df[col].memory_usage(deep=True))
            })
        info["columns"] = column_info
        
        # Dtypes summary
        dtypes_summary = df.dtypes.value_counts().to_dict()
        info["dtypes_summary"] = {str(k): int(v) for k, v in dtypes_summary.items()}
        
        return info
    except Exception as e:
        logging.error(f"Error generating info stats: {e}")
        return {"error": f"Could not generate info stats: {e}"}

def generate_describe(df: pd.DataFrame) -> Dict[str, Any]:
    """Generates descriptive statistics for each column, similar to df.describe()."""
    describe = {}
    try:
        if df.empty:
            return {"message": "DataFrame is empty."}
            
        desc_df = df.describe(include='all').fillna("N/A")
        desc_dict = desc_df.to_dict()
        for col in df.columns:
            if col in desc_dict:
                col_desc = desc_dict[col]
                # Convert values to JSON-serializable types
                col_desc = {k: (float(v) if isinstance(v, (int, float)) and k in ['mean', 'std', 'min', '25%', '50%', '75%', 'max'] else 
                                int(v) if isinstance(v, (int, float)) and k in ['count', 'unique', 'freq'] else 
                                str(v) if k == 'top' or v == "N/A" else v) 
                            for k, v in col_desc.items()}
                describe[col] = col_desc
            else:
                describe[col] = {}
        return describe
    except Exception as e:
        logging.error(f"Error generating describe stats: {e}")
        return {"error": f"Could not generate describe stats: {e}"}

def generate_correlation(df: pd.DataFrame) -> Dict[str, Any]:
    """Generates the correlation matrix for numeric columns."""
    try:
        if df.empty:
            return {"message": "DataFrame is empty."}

        # Select only numeric columns for correlation
        numeric_df = df.select_dtypes(include=np.number)
        if numeric_df.empty:
            return {"message": "No numeric columns to calculate correlation."}
        
        corr_matrix = numeric_df.corr()
        # Fill NaN with null for JSON serialization, then convert to dict
        correlation = json.loads(corr_matrix.to_json(orient='index'))
        return correlation
    except Exception as e:
        logging.error(f"Error generating correlation matrix: {e}")
        return {"error": f"Could not generate correlation matrix: {e}"}

def generate_data_frame_sample(df: pd.DataFrame, num_rows: int = 5) -> List[Dict[str, Any]]:
    """Generates a sample of the DataFrame (head)."""
    try:
        if df.empty:
            return []
        
        sample_json = df.head(num_rows).to_json(orient='records')
        return json.loads(sample_json)
    except Exception as e:
        logging.error(f"Error generating data frame sample: {e}")
        return [{"error": f"Could not generate data frame sample: {e}"}]