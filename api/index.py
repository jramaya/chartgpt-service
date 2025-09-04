# api/index.py
from fastapi import FastAPI

app = FastAPI()

@app.get("/api/hello")
def hello_world():
    return {"message": "Hola desde el backend de Python en Vercel!"}

