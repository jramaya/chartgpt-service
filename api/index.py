# api/index.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from typing import Dict, List, Any
import pandas as pd
import io
import json

app = FastAPI()

@app.get("/api/hello")
def hello_world():
    return {"message": "Hola desde el backend de Python en Vercel!"}

@app.post("/api/read_file")
async def read_file(file: UploadFile = File(...)):
    """Read a CSV or Excel file and return its contents as JSON."""
    filename = file.filename
    try:
        content = await file.read()
        if filename.endswith('.csv'):
            df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        elif filename.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(io.BytesIO(content))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Please upload a CSV or Excel file.")

        return JSONResponse(content={
            "data": json.loads(df.to_json(orient='records')),
            "columns": df.columns.tolist(),
            "shape": df.shape
        })
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/stats")
async def stats(data: Dict[str, Any]):
    """Generate descriptive statistics for the provided data."""
    try:
        # Assume input is a single dict with 'data', 'columns', 'shape'
        if not all(key in data for key in ['data', 'columns', 'shape']):
            raise ValueError("Input must contain 'data', 'columns', and 'shape' keys.")
        
        inner_data = data['data']
        columns = data['columns']
        expected_shape = tuple(data['shape'])
        
        # Create DataFrame based on the type of inner_data
        if isinstance(inner_data, list) and inner_data and isinstance(inner_data[0], dict):
            df = pd.DataFrame(inner_data)
        else:
            df = pd.DataFrame(inner_data, columns=columns)
        
        # Validate shape
        if df.shape != expected_shape:
            raise ValueError(f"Shape mismatch: expected {expected_shape}, got {df.shape}")
        
        # Ensure columns match
        if list(df.columns) != columns:
            df.columns = columns
        
        # Basic info
        class_name = df.__class__.__name__
        shape = list(df.shape)
        
        # Index info
        index_info = {
            "type": str(df.index.__class__.__name__),
            "start": df.index[0] if len(df) > 0 else None,
            "stop": df.index[-1] + 1 if len(df) > 0 and isinstance(df.index, pd.RangeIndex) else None,
            "step": df.index.step if hasattr(df.index, 'step') else 1
        }
        
        # Columns info
        column_info = []
        for col in df.columns:
            non_null = int(df[col].count())
            null_count = len(df) - non_null
            unique = df[col].nunique()
            col_memory = df[col].memory_usage(deep=True)
            column_info.append({
                "name": col,
                "dtype": str(df[col].dtype),
                "non_null": non_null,
                "null": null_count,
                "unique": unique,
                "memory_usage_bytes": int(col_memory)
            })
        
        # Dtypes summary
        dtypes_summary = df.dtypes.value_counts().to_dict()
        dtypes_summary = {str(k): int(v) for k, v in dtypes_summary.items()}
        
        # Descriptive statistics (corregido)
        desc_df = df.describe(include='all').fillna("N/A")
        desc = desc_df.to_dict()
        descriptive_statistics = {}
        for col in df.columns:
            if col in desc:
                col_desc = desc[col]
                # Convertir valores a tipos serializables (e.g., numpy float/int a Python nativo)
                col_desc = {k: (float(v) if isinstance(v, (int, float)) and k in ['mean', 'std', 'min', '25%', '50%', '75%', 'max'] else 
                                int(v) if isinstance(v, (int, float)) and k in ['count', 'unique', 'freq'] else 
                                str(v) if k == 'top' or v == "N/A" else v) 
                            for k, v in col_desc.items()}
                descriptive_statistics[col] = col_desc
            else:
                descriptive_statistics[col] = {}  # En caso de columna sin stats (improbable)
        
        return JSONResponse(content={
            "class": class_name,
            "shape": shape,
            "index": index_info,
            "columns": column_info,
            "dtypes_summary": dtypes_summary,
            "descriptive_statistics": descriptive_statistics
        })
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))