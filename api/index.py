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

@app.post("/api/read_csv")
async def read_csv(file: UploadFile = File(...)):
    """Read a CSV file and return its contents as JSON."""
    try:
        content = await file.read()
        df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        return JSONResponse(content={
            "data": json.loads(df.to_json(orient='records')),
            "columns": df.columns.tolist(),
            "shape": df.shape
        })
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/describe")
async def describe(data: List[Dict[str, Any]]):
    """Generate descriptive statistics for the provided data."""
    try:
        df = pd.DataFrame(data)
        # Use include='all' to get stats for non-numeric columns as well
        description = df.describe(include='all').fillna('N/A')
        return JSONResponse(content={
            "statistics": json.loads(description.to_json()),
            "dtypes": df.dtypes.astype(str).to_dict()
        })
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
