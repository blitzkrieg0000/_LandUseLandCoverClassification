# import necessary libraries
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import numpy as np

# generate some sample data
np.random.seed(42)
df = pd.DataFrame({
    'x': np.arange(1, 101),
    'y': np.random.rand(100) * 100
})

# initialize the Dash app
app = dash.Dash(__name__)

# define the app layout
app.layout = html.Div([
    html.H1("Interactive Scatter Plot with Dash"),
    dcc.Slider(
        id='data-slider',
        min=10,
        max=100,
        step=10,
        value=50,
        marks={i: str(i) for i in range(10, 101, 10)}
    ),
    dcc.Graph(id='scatter-plot')
])

# define the callback to update the scatter plot
@app.callback(
    Output('scatter-plot', 'figure'),
    [Input('data-slider', 'value')]
)
def update_scatter_plot(slider_value):
    # filter the data based on slider value
    filtered_df = df.iloc[:slider_value]
    # create a scatter plot
    fig = px.scatter(filtered_df, x="x", y="y", title=f'Scatter Plot with {slider_value} Data Points')
    return fig

# run the app
if __name__ == '__main__':
    app.run_server(debug=True)
