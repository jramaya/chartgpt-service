# ChartGPT Service

## About The Project

This project is a web application that empowers users to become data analysts instantly. By uploading a spreadsheet (`.xlsx` or `.csv`), the application uses AI to analyze the data, suggest impactful visualizations, and help build a simple, elegant dashboard in seconds.

The goal is to unlock the value trapped in spreadsheets for professionals who lack the time or expertise for complex data exploration, providing instant insights without the need for traditional BI software.

For a more detailed project brief and requirements, please review the **`CHALLENGE.md`** file.

## Built With

- **Frontend**: React
- **Backend**: Python with FastAPI
- **Data Analysis**: Pandas
- **Artificial Intelligence**: OpenAI (GPT)
- **Data Visualization**: ECharts (configurable for others like D3.js)

## Getting Started

Follow these steps to set up and run the project in your local environment.

### Prerequisites

- Python 3.10 or higher
- Node.js and npm (for the React frontend)
- An OpenAI API Key

### 1. Backend Setup (Python/FastAPI)

First, clone the repository and navigate to the project directory:

```bash
git clone <YOUR_REPOSITORY_URL>
cd chartgpt-service
```

Crea y activa un entorno virtual. Usar un entorno virtual es una buena práctica para aislar las dependencias del proyecto.

```bash
python3.10 -m venv venv
source venv/bin/activate
```

Instala las dependencias de Python desde el archivo `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 2. Variables de Entorno

Para que la integración con OpenAI funcione, necesitas configurar tu clave de API. Crea un archivo `.env` en la raíz del proyecto y añade tu clave:

```
OPENAI_API_KEY="tu_clave_secreta_de_openai_aqui"
```

### 3. Ejecutar el Servidor de Desarrollo

Una vez que las dependencias estén instaladas y la variable de entorno configurada, puedes iniciar el servidor de FastAPI. El `--reload` hará que el servidor se reinicie automáticamente cada vez que hagas cambios en el código.

```bash
uvicorn api.index:app --reload
```

El backend ahora estará corriendo en `http://127.0.0.1:8000`.

---
*Nota: Las instrucciones para configurar y ejecutar el frontend de React deberán ser añadidas por separado.*
