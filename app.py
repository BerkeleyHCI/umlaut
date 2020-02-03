import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

import json
import random

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

mock_data = {
    'train': [[0, 1, 2, 3, 4, 5], [12.6, 4.39, 1.1, 0.56, 0.48, 0.44]],
    'val': [[0, 1, 2, 3, 4, 5], [12.6, 6.39, 3.1, 1.56, 1.48, 0.9]],
}

app.layout = html.Div([
    html.Div([
        html.H1('Umlaut Toolkit'),
        html.Hr(),
    ]),
    html.Div([
            html.H3('Loss'),
            html.Button(id='btn-update', n_clicks=0, children='Another one'),
            dcc.Graph(
                id='graph_loss',
                figure={
                    'layout': {'title': 'Loss over Epochs'},
                },
            ),
        ],
        className='five columns',
    ),
    html.Div([
            html.H2('Error Messages'),
            html.Hr(),
        ],
        className='six columns',
    ),
    html.Div(id='cache', style={'display': 'none'}, children=json.dumps(mock_data)),
])

@app.callback(
    Output('cache', 'children'),
    [Input('btn-update', 'n_clicks'),],
    [State('cache', 'children')],
)
def update_mock_data(clicks, mock_data_json):
    mock_data = json.loads(mock_data_json)
    for k in mock_data.keys():
        mock_data[k][0].append(mock_data[k][0][-1] + 1)
        mock_data[k][1].append(mock_data[k][1][-1] * 0.91 + (random.random() - 0.6))
    return json.dumps(mock_data)


@app.callback(
    Output('graph_loss', 'figure'),
    [Input('cache', 'children'),],
)
def update_loss(mock_data_json):
    mock_data = json.loads(mock_data_json)
    return {
        'data': [
            {
                'x': mock_data['train'][0],
                'y': mock_data['train'][1],
                'name': 'train_200203',
                'type': 'line+marker',
            },
            {
                'x': mock_data['val'][0],
                'y': mock_data['val'][1],
                'name': 'val_200203',
                'type': 'line+marker',
            },
        ],
    }


if __name__ == '__main__':
    app.run_server(debug=True)

