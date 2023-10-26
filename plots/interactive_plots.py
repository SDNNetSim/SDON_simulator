import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
import ast


def _filter_data(data, erlang):
    result = {}
    congestion_levels = ['Low', 'Medium', 'High']  # Assuming based on previous discussions

    for idx, array in enumerate(data[erlang]):
        source = idx  # Assuming source is '0' for all these arrays
        for sub_idx, dest_array in enumerate(array):
            if idx == sub_idx:
                continue
            destination = str(sub_idx)  # Using the index as the destination
            source_dest_pair = (f"S{source}", f"D{destination}")
            result[source_dest_pair] = {}

            for i, row in enumerate(dest_array):
                for j, (path, value) in enumerate(row):
                    if path is not None:  # Check if the path is not None
                        path_name = f"Path {i+1}"
                        if path_name not in result[source_dest_pair]:
                            result[source_dest_pair][path_name] = {}
                        result[source_dest_pair][path_name][congestion_levels[j]] = value

    return result


def plot_q_table(data):
    for time, all_sims in data.items():
        for sim_num, sim_obj in all_sims.items():
            erlang_index = -1
            curr_erlang = sim_obj['erlang_vals'][erlang_index]
            filtered_data = _filter_data(data=sim_obj['q_tables'], erlang=erlang_index)
    app = dash.Dash(__name__)

    app.layout = html.Div([
        dcc.Dropdown(
            id='pair-dropdown',
            options=[{'label': f"{s}-{d}", 'value': str((s, d))} for s, d in filtered_data.keys()],
            value=str(list(filtered_data.keys())[0])
        ),
        dcc.Graph(id='heatmap-output')
    ])

    @app.callback(
        Output('heatmap-output', 'figure'),
        [Input('pair-dropdown', 'value')]
    )
    def update_heatmap(pair_str):
        s, d = ast.literal_eval(pair_str)
        df = pd.DataFrame(filtered_data[(s, d)])
        fig = go.Figure(data=go.Heatmap(z=df, x=df.columns, y=df.index, colorscale="YlGnBu"))
        fig.update_layout(
            title=f"Q-values for {s}-{d} pair (E={curr_erlang})",
            xaxis=dict(
                title_text="Paths",
                title_font=dict(family="Arial", size=18, color="black"),
                tickfont=dict(family="Arial", size=16, color="black")
            ),
            yaxis=dict(
                title_text="Congestion Levels",
                title_font=dict(family="Arial", size=18, color="black"),
                tickfont=dict(family="Arial", size=16, color="black")
            )
        )
        return fig

    app.run_server(debug=True)
