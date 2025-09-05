# api/index.py
from fastapi import FastAPI, UploadFile, File, HTTPException, Body, Query
from fastapi.responses import JSONResponse
from fastapi_mcp import FastApiMCP
from typing import List, Any
from pydantic import BaseModel
from enum import Enum
import pandas as pd
import io
import json
import os
import openai
import logging

from src.utils.execute_pandas import execute_pandas_operation
from src.utils.stats import (
    generate_info,
    generate_describe,
    generate_correlation,
    generate_data_frame_sample
)
from src.utils.prompt_generators import (
    get_system_prompt_for_backend
)

# --- Pydantic Models for Request Validation ---

class ChartConfiguration(BaseModel):
    id: str | int
    title: str
    pandas_operation: str

class DataFramePayload(BaseModel):
    data: List[dict]
    columns: List[str]
    shape: List[int]

# --- Pydantic Models for Stats Payload ---

class InfoIndex(BaseModel):
    type: str
    start: int
    stop: int
    step: int

class InfoColumn(BaseModel):
    name: str
    dtype: str
    non_null: int
    null: int
    unique: int
    memory_usage_bytes: int

class InfoPayload(BaseModel):
    class_: str | None = None
    shape: List[int]
    index: InfoIndex
    columns: List[InfoColumn]
    dtypes_summary: dict[str, int]

class DataFrameSamplePayload(BaseModel):
    data: List[dict]
    columns: List[str]
    shape: List[int]

class StatsPayload(BaseModel):
    info: InfoPayload | dict # Allow dict for flexibility with raw JSON
    describe: dict
    correlation: dict
    dataFrameSample: List[dict] | DataFrameSamplePayload # Allow both formats

class DataFramePandasOperation(BaseModel):
    data_frame: DataFramePayload
    operations: List[ChartConfiguration]

class GenerateChartsConfigurationsResponse(BaseModel):
    operations: List[ChartConfiguration]

class OperationResult(ChartConfiguration):
    result: Any

class BuildChartsResponse(BaseModel):
    results: List[OperationResult | ChartConfiguration] # Allow both for define_charts_template

# --- Enums for Query Parameters ---
class ChartsBackend(str, Enum):
    echarts = "echarts"
    d3 = "d3"

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
    "/api/generate_charts_configurations",
    operation_id="generate_charts_configurations",
    response_model=GenerateChartsConfigurationsResponse,
    summary="Generates chart configurations from data statistics",
    description="Receives DataFrame statistics and uses an AI model to generate a list of chart configurations for a specified backend (e.g., ECharts). The AI suggests insightful visualizations based on the data summary."
)
async def generate_charts_configurations(
    payload: StatsPayload,
    charts_backend: ChartsBackend = Query(
        default=ChartsBackend.echarts,
        description="The charting library backend to generate configurations for."
    )
):
    """
    This endpoint uses an AI to generate chart configurations.
    
    - It receives a payload with statistics about a DataFrame.
    - It takes a `charts_backend` query parameter to specify the desired charting library.
    - It constructs a system prompt tailored to the selected backend.
    - It calls an AI model (e.g., OpenAI's GPT) to generate a list of operations.
    - Each operation contains the `pandas_operation` needed to produce a chart configuration
      (like an ECharts option object) that can be executed by the `/api/build_charts` endpoint.
    """
    if not client:
        raise HTTPException(
            status_code=503,
            detail="OpenAI client is not configured. Please set the OPENAI_API_KEY environment variable."
        )

    try:
        system_prompt = get_system_prompt_for_backend(charts_backend.value)
    except NotImplementedError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    # Define the tool for the AI to use (function calling)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "create_chart_operation",
                "description": "Generates a single, insightful chart suggestion based on the provided data summary. This function will be called multiple times to suggest several different charts.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "A concise and descriptive title for the chart (e.g., 'CO2 Emissions vs. Vehicle Weight')."
                        },
                        "pandas_operation": {
                            "type": "string",
                            "description": "A string of Python code that will be executed to generate an ECharts configuration dictionary. The code has access to a pandas DataFrame called `df`."
                        },
                        "insight": {
                            "type": "string",
                            "description": "A brief, insightful comment about what the chart reveals from the data."
                        }
                    },
                    "required": ["title", "pandas_operation", "insight"]
                }
            }
        }
    ]

    user_prompt = f"""
    Here is a summary of the data to analyze:
    Info: {json.dumps(payload.info if isinstance(payload.info, dict) else payload.info.model_dump(), indent=2)}
    Describe (Descriptive Statistics): {json.dumps(payload.describe, indent=2)}
    Correlation Matrix: {json.dumps(payload.correlation, indent=2)}
    Data Sample (first 5 rows): {json.dumps(payload.dataFrameSample if isinstance(payload.dataFrameSample, list) else payload.dataFrameSample.model_dump(), indent=2)}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            tools=tools,
            tool_choice="auto",
        )

        operations = []
        for tool_call in response.choices[0].message.tool_calls or []:
            args = json.loads(tool_call.function.arguments)
            pandas_operation = args['pandas_operation']
            operations.append({"id": tool_call.id, "title": args['title'], "pandas_operation": pandas_operation})

        return {"operations": operations}
    except Exception as e:
        logging.error(f"Error calling OpenAI or processing response: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate chart operations from AI.")


@app.post(
    "/api/build_charts",
    operation_id="build_charts",
    response_model=BuildChartsResponse,
    summary="Executes pandas code to generate chart configurations",
    description="This endpoint receives a DataFrame and a list of pandas code operations. It executes each operation to transform data or generate chart configurations (e.g., for ECharts). It returns the results of each operation, ideal for building a dynamic dashboard."
)
async def build_charts(payload: DataFramePandasOperation = Body(
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
                "pandas_operation": "{'title': {'text': 'Weight vs CO2 Emissions'}, 'xAxis': {'type': 'value', 'name': 'Weight (kg)'}, 'yAxis': {'type': 'value', 'name': 'CO2 (g/km)'}, 'series': [{'type': 'scatter', 'data': df[['Weight', 'CO2']].values.tolist()}]}"
            },
            {
                "id": "correlation_matrix",
                "title": "Correlation Matrix",
                "pandas_operation": "df[['Volume', 'Weight', 'CO2']].corr()"
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
        # Convert Pydantic models to dicts for execute_pandas_operation
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
                result = execute_pandas_operation(df.copy(), op)
                results.append(result)
            except Exception as e:
                logging.warning(f"Skipping operation {op.get('id', 'N/A')} due to an error: {e}")

        return {"results": results}

    except Exception as e:
        logging.error(f"An unexpected error occurred in build_charts: {e}")
        raise HTTPException(status_code=400, detail=str(e))