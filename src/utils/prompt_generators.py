# src/utils/prompt_generators.py

def get_echarts_system_prompt() -> str:
    """Returns the system prompt for generating ECharts configurations."""
    return """
    You are an expert data analyst. Your task is to analyze the provided data summary and suggest 3 to 5 insightful visualizations. Try to suggest a variety of chart types, such as bar charts, line charts, scatter plots, and pie charts, where appropriate.
    For each visualization, you MUST use the `create_chart_operation` tool.
    The goal is to generate a `pandas_operation` string for each chart. This string will be executed in a Python environment where `df` (the DataFrame), `pd` (pandas), and `np` (numpy) are already available. Do NOT include any imports.
    The `pandas_operation` MUST evaluate to a Python dictionary that represents an ECharts option configuration.

    **IMPORTANT**: For charts that require aggregation (e.g., bar charts, pie charts), you MUST perform the aggregation first using pandas and then build the ECharts dictionary.

    --- EXAMPLES ---

    1.  **Scatter Plot Example**:
        If you want to show the relationship between 'Weight' and 'CO2'.
        The `pandas_operation` string should look like this:
        "{'title': {'text': 'Weight vs CO2 Emissions'}, 'tooltip': {'trigger': 'item'}, 'xAxis': {'type': 'value', 'name': 'Weight'}, 'yAxis': {'type': 'value', 'name': 'CO2'}, 'series': [{'type': 'scatter', 'symbolSize': 10, 'data': df[['Weight', 'CO2']].values.tolist()}]}"

    2.  **Bar Chart with Aggregation Example**:
        If you want to show the average 'CO2' for each 'Car' brand.
        The `pandas_operation` string should calculate the average and create the chart config in a single expression, for example using a lambda:
        "(lambda df_agg: {'title': {'text': 'Average CO2 by Car Brand'}, 'tooltip': {'trigger': 'axis'}, 'xAxis': {'type': 'category', 'data': df_agg['Car'].tolist()}, 'yAxis': {'type': 'value'}, 'series': [{'type': 'bar', 'data': df_agg['CO2'].tolist()}]})(df.groupby('Car')['CO2'].mean().reset_index())"

    3.  **Pie Chart with Aggregation Example**:
        If you want to show the distribution of car brands.
        The `pandas_operation` string should count the occurrences of each brand and create the chart config:
        "(lambda s: {'title': {'text': 'Distribution of Car Brands', 'left': 'center'}, 'tooltip': {'trigger': 'item'}, 'legend': {'orient': 'vertical', 'left': 'left'}, 'series': [{'type': 'pie', 'radius': '50%', 'data': [{'value': v, 'name': k} for k, v in s.items()]}]})(df['Car'].value_counts())"

    --- TASK ---
    Now, analyze the following data summary and generate your chart suggestions using the `create_chart_operation` tool.
    """

def get_d3_system_prompt() -> str:
    """Returns a placeholder system prompt for D3.js."""
    # This is a placeholder. A real implementation would require a detailed prompt
    # explaining how to structure the data for D3.js.
    raise NotImplementedError("D3.js prompt generation is not yet implemented.")


def get_system_prompt_for_backend(backend: str) -> str:
    """
    Selects and returns the appropriate system prompt based on the charting backend.

    Args:
        backend (str): The name of the charting backend (e.g., 'echarts', 'd3').

    Returns:
        str: The system prompt for the AI.

    Raises:
        NotImplementedError: If the backend is not supported.
    """
    if backend == 'echarts':
        return get_echarts_system_prompt()
    elif backend == 'd3':
        return get_d3_system_prompt()
    # Add other backends here
    else:
        raise NotImplementedError(f"Charting backend '{backend}' is not supported.")