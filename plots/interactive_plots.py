import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
import ast

# Sample data for multiple source-destination pairs
data = {
    ('S1', 'D1'): {
        'Path 1': {'Low': 5, 'Medium': 10, 'High': 2},
        'Path 2': {'Low': 3, 'Medium': 8, 'High': 7},
        'Path 3': {'Low': 6, 'Medium': 5, 'High': 3},
    },
    ('S2', 'D2'): {
        'Path 1': {'Low': 5, 'Medium': 10, 'High': 2},
        'Path 2': {'Low': 300, 'Medium': 0, 'High': 1},
        'Path 3': {'Low': 632, 'Medium': 53, 'High': 3},
    },
    ('S3', 'D3'): {
        'Path 1': {'Low': 5, 'Medium': 1033, 'High': 2},
        'Path 2': {'Low': 333, 'Medium': 8, 'High': 7},
        'Path 3': {'Low': 699, 'Medium': 5, 'High': 3},
    },
}

# Initialize the app
app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Dropdown(
        id='pair-dropdown',
        # Using str() to represent the tuple as a string for the dropdown values
        options=[{'label': f"{s}-{d}", 'value': str((s, d))} for s, d in data.keys()],
        value=str(list(data.keys())[0])
    ),
    dcc.Graph(id='heatmap-output')
])


@app.callback(
    Output('heatmap-output', 'figure'),
    [Input('pair-dropdown', 'value')]
)
def update_heatmap(pair_str):
    # Convert the string representation of the tuple back to an actual tuple
    s, d = ast.literal_eval(pair_str)
    df = pd.DataFrame(data[(s, d)])
    fig = go.Figure(data=go.Heatmap(z=df, x=df.columns, y=df.index, colorscale="YlGnBu"))
    fig.update_layout(
        title=f"Q-values for {s}-{d} pair",
        xaxis=dict(
            title_text="Paths",
            title_font=dict(family="Arial", size=18, color="black"),  # Adjust size as needed
            tickfont=dict(family="Arial", size=16, color="black")  # Adjust size as needed
        ),
        yaxis=dict(
            title_text="Congestion Levels",
            title_font=dict(family="Arial", size=18, color="black"),  # Adjust size as needed
            tickfont=dict(family="Arial", size=16, color="black")  # Adjust size as needed
        )
    )
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
