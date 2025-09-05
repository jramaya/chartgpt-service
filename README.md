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
git clone https://github.com/jramaya/chartgpt-service.git
cd chartgpt-service
```
Create and activate a virtual environment. Using a virtual environment is a good practice to isolate project dependencies.
```bash
python3.10 -m venv venv
source venv/bin/activate
```
Install the Python dependencies from the `requirements.txt` file:
```bash
pip install -r requirements.txt
```
### 2. Environment Variables
For the OpenAI integration to work, you need to set up your API key. Create a `.env` file in the project root and add your key:
```
OPENAI_API_KEY="your_openai_secret_key_here"
```
### 3. Run the Development Server
Once the dependencies are installed and the environment variable is set, you can start the FastAPI server. The `--reload` flag will make the server restart automatically whenever you make changes to the code.
```bash
uvicorn api.index:app --reload
```
The backend will now be running at `http://127.0.0.1:8000`.
---
*Note: Instructions for setting up and running the React frontend will need to be added separately.*
