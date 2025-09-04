# api/index.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from typing import List, Any
from pydantic import BaseModel
import pandas as pd
import io
import json
import logging

from src.utils.execute_pandas import execute_pandas_code
from src.utils.stats import (
    generate_info,
    generate_describe,
    generate_correlation,
    generate_data_frame_sample
)

# --- Pydantic Models for Request Validation ---

class PandasOperation(BaseModel):
    id: str | int
    title: str
    pandas_code: str

class DataFramePayload(BaseModel):
    data: List[Any]
    columns: List[str]
    shape: List[int]

class ExecuteCodePayload(BaseModel):
    data_frame: DataFramePayload
    operations: List[PandasOperation]

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
async def stats(payload: DataFramePayload):
    """Generate descriptive statistics for the provided data."""
    try:
        inner_data = payload.data
        columns = payload.columns
        expected_shape = tuple(payload.shape)
        
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
        
        return JSONResponse(content={
            "info": generate_info(df),
            "describe": generate_describe(df),
            "correlation": generate_correlation(df),
            "dataFrameSample": generate_data_frame_sample(df)
        })
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))