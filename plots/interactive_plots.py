import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
import ast


def _filter_data(data, erlang):
    """
    This function takes a dictionary of Erlang-related data and a specific key representing Erlang values. It filters and organizes the data to create a new dictionary with source-destination pairs as keys. The values associated with each pair include information about congestion levels on different paths.
    The function iterates through the provided data, extracting relevant information such as source, destination, and congestion levels. It checks for non-None paths and structures the result accordingly. The congestion levels are categorized as 'Low', 'Medium', and 'High'.

    :param data: A dictionary containing Erlang-related data.
    :param erlang: The key representing the Erlang data to be processed.

    :return: A filtered and organized dictionary with source-destination pairs as keys and congestion level information.
    :rtype: dict
    """
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
    """
    Generate and display a Dash web application with a dropdown menu and heatmap for Q-values.

    :param-data: A nested dictionary containing simulation data
    
    """
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
        """
        This function takes a string representation of a tuple containing source and destination information (`pair_str`) and generates a heatmap of Q-values for the corresponding source-destination pair.

        :param pair_str: A string representation of a tuple containing source and destination information.
        :type pair_str: str

        :return: A Plotly figure representing a heatmap of Q-values for the specified source-destination pair.

        :rtype: plot
        """
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
