# api/index.py
from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.responses import JSONResponse
from fastapi_mcp import FastApiMCP
from typing import List, Any
from pydantic import BaseModel
import pandas as pd
import io
import json
import os
import openai
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

class StatsPayload(BaseModel):
    info: str
    describe: dict
    correlation: dict
    dataFrameSample: dict

class ExecuteCodePayload(BaseModel):
    data_frame: DataFramePayload
    operations: List[PandasOperation]

class GeneratePandasOperationsResponse(BaseModel):
    operations: List[PandasOperation]

class OperationResult(PandasOperation):
    result: Any

class BuildChartsResponse(BaseModel):
    results: List[OperationResult | PandasOperation] # Allow both for define_charts_template

# --- OpenAI Client Setup ---
# It's recommended to set the OPENAI_API_KEY as an environment variable
if os.getenv("OPENAI_API_KEY"):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    client = openai.OpenAI()
else:
    client = None
    logging.warning("OPENAI_API_KEY environment variable not set. OpenAI-related endpoints will not work.")

app = FastAPI()
# mcp = FastApiMCP(app,
#     name="Instant Analysis MCP",
#     description="MCP server for data analysis and visualization generation.",
#     describe_all_responses=True,
#     describe_full_response_schema=True,
#     include_operations=["define_charts_template", "read_file", "stats"]
#                  )

# Mount the MCP server
# mcp.mount()

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
    "/api/generate_pandas_operations",
    operation_id="generate_pandas_operations",
    response_model=GeneratePandasOperationsResponse,
    summary="Generates a template of pandas operations for charts",
    description="Receives DataFrame statistics and returns a sample list of pandas operations. This endpoint acts as a placeholder, demonstrating the structure required by the `/api/build_charts` endpoint. The input payload is validated but not used in the current template implementation."
)
async def generate_pandas_operations(payload: StatsPayload = Body(
    ...,
    example={
        "info": "RangeIndex: 36 entries, 0 to 35\nData columns (total 4 columns):\n #   Column  Non-Null Count  Dtype  \n---  ------  --------------  -----  \n 0   Car     36 non-null     object \n 1   Volume  36 non-null     float64\n 2   Weight  36 non-null     int64  \n 3   CO2     36 non-null     int64  \ndtypes: float64(1), int64(2), object(1)",
        "describe": {
            "Volume": {"count": 36.0, "mean": 1.611111, "std": 0.388975, "min": 1.0, "25%": 1.275, "50%": 1.6, "75%": 2.0, "max": 2.5},
            "Weight": {"count": 36.0, "mean": 1292.277778, "std": 240.145928, "min": 790.0, "25%": 1117.5, "50%": 1329.0, "75%": 1482.5, "max": 1746.0},
            "CO2": {"count": 36.0, "mean": 104.027778, "std": 7.454531, "min": 90.0, "25%": 99.0, "50%": 105.0, "75%": 111.25, "max": 120.0}
        },
        "correlation": {
            "Volume": {"Volume": 1.0, "Weight": 0.758112, "CO2": 0.592082},
            "Weight": {"Volume": 0.758112, "Weight": 1.0, "CO2": 0.552152},
            "CO2": {"Volume": 0.592082, "Weight": 0.552152, "CO2": 1.0}
        },
        "dataFrameSample": {
            "data": [{"Car": "Toyoty", "Volume": 1.0, "Weight": 790, "CO2": 90}],
            "columns": ["Car", "Volume", "Weight", "CO2"],
            "shape": [1, 4]
        }
    }
)):
    """
    This endpoint returns a hardcoded list of example pandas operations.
    
    It serves as a placeholder or template, demonstrating the data structure
    that would be generated by an AI and is expected by the `/api/build_charts` endpoint.
    Each operation in the list is a dictionary containing:
    - `id`: A unique identifier for the chart.
    - `title`: A descriptive title for the chart.
    - `pandas_code`: A string of Python code that generates a chart configuration.
    """
    # This is a placeholder implementation.
    # It ignores the payload and returns a hardcoded list of operations as a template.
    example_operations = [
        {
            "id": "echarts_scatter_weight_co2",
            "title": "Scatter Plot: Weight vs CO2",
            "pandas_code": "{'title': {'text': 'Weight vs CO2 Emissions'}, 'xAxis': {'type': 'value', 'name': 'Weight (kg)'}, 'yAxis': {'type': 'value', 'name': 'CO2 (g/km)'}, 'series': [{'type': 'scatter', 'data': df[['Weight', 'CO2']].values.tolist()}]}"
        }
    ]
    return {"operations": example_operations}


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