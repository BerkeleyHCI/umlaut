import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

import json
import random

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# ----------- Mock Data ---------- #

mock_data = {
    'train': [[0, 1, 2, 3, 4, 5], [12.6, 4.39, 1.1, 0.56, 0.48, 0.44]],
    'val': [[0, 1, 2, 3, 4, 5], [12.6, 6.39, 3.1, 1.56, 1.48, 0.9]],
}

mock_error_msgs = [
    {
        'epoch': 3,
        'title': '⚠️ This demo is fake.',
        'description': '''This demo is fake. Unfortunately what you're seeing right
    now is a mockup of the final interface. This usually happens when you are early
    in the design process and want to validate your decisions before committing to
    a more significant engineering effort. You can solve this problem by writing
    code in your editor like: 

```py
[f'element {e} number {i}' for i, e in enumerate(range(10)) if valid(e)]
```''',
    }, {
        'epoch': 4,
        'title': 'This is another error message.',
        'description': 'it works, lol.',
    },
]

# ---------- Helper Functions for Rendering ---------- #


def render_error_messages(errors_data):
    result_divs = []
    for i, error in enumerate(errors_data):
        result_divs.append(render_error_message('error-msg-{}'.format(i), error))
    return result_divs


def render_error_message(id, error_message):
    return html.Div([
        html.H3(error_message['title']),
        dcc.Markdown(error_message['description']),
        html.Small('Captured at epoch {}.'.format(error_message['epoch'])),
        html.Hr(),
    ], id=id, style={'display': 'inline-block'})


# ----------- App Layout ---------- #


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
            dcc.Graph(
                id='graph_acc',
                figure={
                    'layout': {
                        'title': 'Accuracy over Epochs',
                    },
                },
            ),
        ],
        className='five columns',
    ),
    html.Div([
            html.H2('Error Messages'),
            html.Hr(),
            html.Div(
                render_error_messages(mock_error_msgs),
                id='errors-list',
            ),
        ],
        className='six columns',
    ),
    html.Div(id='cache', style={'display': 'none'}, children=json.dumps(mock_data)),
])


# ---------- App Callbacks ---------- #

@app.callback(
    Output('cache', 'children'),
    [Input('btn-update', 'n_clicks')],
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
        'layout': {
            'title': 'Loss over epochs',
            'shapes': [{
                'type': 'rect',
                'xref': 'x',
                'yref': 'paper',
                'x0': 3, # x0, x1 are epoch bounds
                'x1': 5,
                'y0': 0,
                'y1': 1,
                'fillcolor': 'LightSalmon',
                'opacity': 0.5,
                'layer': 'below',
                'line_width': 0,
            }],
        },
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

