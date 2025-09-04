# api/index.py
from fastapi import FastAPI, UploadFile, File, HTTPException
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

class ExecuteCodePayload(BaseModel):
    data_frame: DataFramePayload
    operations: List[PandasOperation]

app = FastAPI()
mcp = FastApiMCP(app,
    name="Instant Analysis MCP",
    description="MCP server for data analysis and visualization generation.",
    describe_all_responses=True,
    describe_full_response_schema=True,
    include_operations=["stats", "build_charts"]
                 )

# Mount the MCP server
mcp.mount()

@app.get("/api/hello")
def hello_world():
    return {"message": "Hola desde el backend de Python en Vercel!"}

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

@app.post("/api/build_charts", operation_id="build_charts")
async def build_charts(payload: ExecuteCodePayload):
    """
    Receives a DataFrame representation and a list of pandas code operations to execute and build a chart.

    The `operations` part of the payload should be a list of objects, where each object represents
    a pandas operation to be executed on the DataFrame. For example:

    "operations": [
        {
            "id": "filter_low_co2",
            "title": "Filtrar carros con CO2 menor a 100",
            "pandas_code": "df[df['CO2'] < 100]"
        },
        {
            "id": "correlation_matrix",
            "title": "Matriz de correlación entre Volume, Weight y CO2",
            "pandas_code": "df[['Volume', 'Weight', 'CO2']].corr()"
        },
        {
            "id": "sort_by_weight",
            "title": "Ordenar por peso descendente",
            "pandas_code": "df.sort_values(by='Weight', ascending=False)"
        },
        {
            "id": "echarts_scatter_weight_co2",
            "title": "Gráfico de dispersión: Peso vs CO2",
            "pandas_code": "{'title': {'text': 'Peso vs Emisiones de CO2'}, 'xAxis': {'type': 'value', 'name': 'Peso (kg)'}, 'yAxis': {'type': 'value', 'name': 'CO2 (g/km)'}, 'series': [{'type': 'scatter', 'data': df[['Weight', 'CO2']].values.tolist()}]}"
        },
        {
            "id": "echarts_line_volume_co2",
            "title": "Gráfico de línea: Volumen vs CO2 ordenado",
            "pandas_code": "{'title': {'text': 'Volumen vs CO2 (Ordenado por Volumen)'}, 'xAxis': {'type': 'category', 'data': (sorted_df := df.sort_values('Volume'))['Volume'].astype(str).tolist()}, 'yAxis': {'type': 'value'}, 'series': [{'type': 'line', 'data': sorted_df['CO2'].tolist()}]}"
        },
        {
            "id": "echarts_pie_brand_count",
            "title": "Gráfico de pastel: Conteo por marca",
            "pandas_code": "{'title': {'text': 'Distribución de Marcas de Autos'}, 'tooltip': {'trigger': 'item'}, 'series': [{'type': 'pie', 'radius': '50%', 'data': [{'value': count, 'name': brand} for brand, count in df['Car'].value_counts().items()]}]}"
        },
        {
            "id": "echarts_histogram_co2",
            "title": "Histograma: Distribución de CO2",
            "pandas_code": "{'title': {'text': 'Distribución de Emisiones de CO2'}, 'xAxis': {'type': 'category', 'data': (bins := np.histogram_bin_edges(df['CO2'], bins=10).astype(int).tolist())}, 'yAxis': {'type': 'value'}, 'series': [{'type': 'bar', 'data': np.histogram(df['CO2'], bins=10)[0].tolist()}]}"
        },
        {
            "id": "echarts_heatmap_correlation",
            "title": "Mapa de calor: Matriz de correlación",
            "pandas_code": "{'title': {'text': 'Matriz de Correlación'}, 'xAxis': {'type': 'category', 'data': (corr := df[['Volume', 'Weight', 'CO2']].corr()).columns.tolist()}, 'yAxis': {'type': 'category', 'data': corr.index.tolist()}, 'visualMap': {'min': -1, 'max': 1, 'calculable': True}, 'series': [{'type': 'heatmap', 'data': [[i, j, corr.iloc[i, j]] for i in range(len(corr)) for j in range(len(corr))]}]}"
        }
    ]
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

        return JSONResponse(content={"results": results})

    except Exception as e:
        logging.error(f"An unexpected error occurred in build_charts: {e}")
        raise HTTPException(status_code=400, detail=str(e))