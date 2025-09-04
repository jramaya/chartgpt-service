# api/index.py
from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.responses import JSONResponse
from fastapi_mcp import FastApiMCP
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

class DataFrameSchemaPayload(BaseModel):
    columns: List[str]
    shape: List[int]

class DefineChartsPayload(BaseModel):
    schema: DataFrameSchemaPayload
    operations: List[PandasOperation]

class ExecuteCodePayload(BaseModel):
    data_frame: DataFramePayload
    operations: List[PandasOperation]

class DefineChartsResponse(BaseModel):
    data_frame: DataFramePayload | None = None # Placeholder
    operations: List[PandasOperation]

class OperationResult(PandasOperation):
    result: Any

class BuildChartsResponse(BaseModel):
    results: List[OperationResult | PandasOperation] # Allow both for define_charts


app = FastAPI()
mcp = FastApiMCP(app,
    name="Instant Analysis MCP",
    description="MCP server for data analysis and visualization generation.",
    describe_all_responses=True,
    describe_full_response_schema=True,
    include_operations=["define_charts", "read_file", "stats"]
                 )

# Mount the MCP server
mcp.mount()

@app.get("/api/ping")
def ping():
    return {"ping ": "From Instant Analysis MCP"}

@app.post("/api/read_file", operation_id="read_file")
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

@app.post("/api/stats", operation_id="stats")
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

@app.post(
    "/api/define_charts",
    operation_id="define_charts",
    response_model=DefineChartsResponse,
    summary="Defines pandas code operations to generate chart configurations",
    description="This endpoint receives a DataFrame schema and a list of pandas code operations. It validates the structure and returns a payload template with the same operations, ready to be sent to /api/build_charts after adding the data. It's ideal for preparing a dynamic dashboard build request."
)
async def define_charts(payload: DefineChartsPayload = Body(
    ...,
    example={
        "schema": {
            "columns": ["Car", "Volume", "Weight", "CO2"],
            "shape": [36, 4]
        },
        "operations": [
            {
                "id": "echarts_scatter_weight_co2",
                "title": "Scatter Plot: Weight vs CO2",
                "pandas_code": "{'title': {'text': 'Weight vs CO2 Emissions'}, 'xAxis': {'type': 'value', 'name': 'Weight (kg)'}, 'yAxis': {'type': 'value', 'name': 'CO2 (g/km)'}, 'series': [{'type': 'scatter', 'data': df[['Weight', 'CO2']].values.tolist()}]}"
            }
        ]
    }
)):
    """
    Receives a DataFrame schema and a list of pandas code operations to define a chart build request.

    The `operations` part of the payload should be a list of objects, where each object represents
    a pandas operation to be executed later by the `build_charts` endpoint.

    The code can either transform the DataFrame (e.g., filtering, sorting) or return a dictionary
    that represents a chart configuration (e.g., for ECharts).

    - **For DataFrame transformations**, the result will be a JSON representation of the resulting DataFrame.
    - **For chart configurations**, the result will be the JSON object itself.

    Example Operations:
    - `df[df['CO2'] < 100]` (Filters the DataFrame)
    - `df.corr()` (Calculates correlation matrix)
    - `{'title': {'text': 'Weight vs CO2'}, 'series': [{'type': 'scatter', 'data': df[['Weight', 'CO2']].values.tolist()}]}` (Creates an ECharts scatter plot config)
    """
    return {
        # The data_frame is returned as null. The client should replace this with the actual DataFrame.
        "data_frame": None,
        "operations": payload.operations
    }


@app.post(
    "/api/build_charts",
    operation_id="build_charts",
    response_model=BuildChartsResponse,
    summary="Executes pandas code to generate chart configurations",
    description="This endpoint receives a DataFrame and a list of pandas code operations. It executes each operation to transform data or generate chart configurations (e.g., for ECharts). It returns the results of each operation, ideal for building a dynamic dashboard."
)
async def build_charts(payload: ExecuteCodePayload = Body(
    ...,
    example={
        "data_frame": {
            "data": [
                {"Car": "Audi", "Volume": 1.6, "Weight": 1250, "CO2": 105},
                {"Car": "BMW", "Volume": 2.0, "Weight": 1350, "CO2": 115},
                {"Car": "Volvo", "Volume": 2.0, "Weight": 1450, "CO2": 120},
                {"Car": "Ford", "Volume": 1.5, "Weight": 1150, "CO2": 99},
            ],
            "columns": ["Car", "Volume", "Weight", "CO2"],
            "shape": [4, 4]
        },
        "operations": [
            {
                "id": "echarts_scatter_weight_co2",
                "title": "Scatter Plot: Weight vs CO2",
                "pandas_code": "{'title': {'text': 'Weight vs CO2 Emissions'}, 'xAxis': {'type': 'value', 'name': 'Weight (kg)'}, 'yAxis': {'type': 'value', 'name': 'CO2 (g/km)'}, 'series': [{'type': 'scatter', 'data': df[['Weight', 'CO2']].values.tolist()}]}"
            },
            {
                "id": "correlation_matrix",
                "title": "Correlation Matrix",
                "pandas_code": "df[['Volume', 'Weight', 'CO2']].corr()"
            }
        ]
    }
)):
    """
    Receives a DataFrame representation and a list of pandas code operations to execute and build a chart.

    The `operations` part of the payload should be a list of objects, where each object represents
    a pandas operation to be executed on the DataFrame.

    The code can either transform the DataFrame (e.g., filtering, sorting) or return a dictionary
    that represents a chart configuration (e.g., for ECharts).

    - **For DataFrame transformations**, the result will be a JSON representation of the resulting DataFrame.
    - **For chart configurations**, the result will be the JSON object itself.

    Example Operations:
    - `df[df['CO2'] < 100]` (Filters the DataFrame)
    - `df.corr()` (Calculates correlation matrix)
    - `{'title': {'text': 'Weight vs CO2'}, 'series': [{'type': 'scatter', 'data': df[['Weight', 'CO2']].values.tolist()}]}` (Creates an ECharts scatter plot config)
    """
    try:
        df_data = payload.data_frame
        # Convert Pydantic models to dicts for execute_pandas_code
        operations = [op.model_dump() for op in payload.operations]

        # Recreate DataFrame from payload
        inner_data = df_data.data
        columns = df_data.columns
        if isinstance(inner_data, list) and inner_data and isinstance(inner_data[0], dict):
            df = pd.DataFrame(inner_data)
        else:
            df = pd.DataFrame(inner_data, columns=columns)

        # Execute the pandas code for each operation. If an operation fails, it's skipped.
        results = []
        for op in operations:
            try:
                # Pass a copy of the DataFrame to prevent side effects between operations
                result = execute_pandas_code(df.copy(), op)
                results.append(result)
            except Exception as e:
                logging.warning(f"Skipping operation {op.get('id', 'N/A')} due to an error: {e}")

        return {"results": results}

    except Exception as e:
        logging.error(f"An unexpected error occurred in build_charts: {e}")
        raise HTTPException(status_code=400, detail=str(e))