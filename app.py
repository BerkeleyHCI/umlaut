# -*- coding: utf-8 -*-

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

import json
import random

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(
    __name__,
    external_stylesheets=external_stylesheets,
    suppress_callback_exceptions=True,  # must have this for dynamic callbacks
)

def argmax(l):
    return max(range(len(l)), key=lambda i: l[i])

# ----------- Mock Data ---------- #

mock_data = {
    'loss': {
        'train': [[0, 1, 2, 3, 4, 5], [12.6, 4.39, 1.1, 0.56, 0.48, 0.44]],
        'val': [[0, 1, 2, 3, 4, 5], [12.6, 6.39, 3.1, 1.56, 1.48, 0.9]],
    },
    'acc': {
        'train': [[0, 1, 2, 3, 4, 5], [10.6, 30.39, 51.1, 60.56, 65.48, 70.44]],
        'val': [[0, 1, 2, 3, 4, 5], [12.6, 26.39, 43.1, 41.56, 45.48, 46.9]],
    }
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
        'annotations': [2, 3],  # right now this will just annotate all plots
    }, {
        'epoch': 5,
        'title': 'This is another error message.',
        'description': 'it works, lol.',
        'annotations': [4, 5],
    },
]


# ----------- App Layout ---------- #

app.layout = html.Div([
    html.Div([
        html.H1('Umlaut Toolkit'),
        html.Hr(),
    ]),
    html.Div([
            html.H3('Loss'),
            html.Button(id='btn-update', n_clicks=0, children='Another one'),
            html.Button(id='btn-update-errors', n_clicks=0, children='Render errors'),
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
            html.Div(id='errors-list'),
        ],
        className='six columns',
    ),
    dcc.Store(id='metrics-cache', storage_type='memory'),
    dcc.Store(id='errors-cache', storage_type='memory'),
    dcc.Store(id='annotations-cache', storage_type='memory'),
])


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
    ], id=id, style={'display': 'inline-block'}, key=id)


# ---------- App Callbacks ---------- #

@app.callback(
    Output('annotations-cache', 'data'),
    [Input('error-msg-{}'.format(i), 'n_clicks_timestamp') for i in range(len(mock_error_msgs))],
)
def highlight_graph_for_error(*click_timestamps):
    click_timestamps = [i or 0 for i in click_timestamps]
    clicked_idx = argmax(click_timestamps)
    annotations = mock_error_msgs[clicked_idx]['annotations']
    return annotations
            

@app.callback(
    Output('metrics-cache', 'data'),
    [Input('btn-update', 'n_clicks')],
    [State('metrics-cache', 'data')],
)
def update_metrics_data(clicks, metrics_data):
    '''handle updates to the metrics data'''
    if clicks is None:
        raise PreventUpdate
    elif clicks == 0:
        return mock_data  # here we have a default value instead of nuthin

    if 'selected' in metrics_data:
        del metrics_data['selected']

    for k in metrics_data.keys():
        metrics_data[k][0].append(metrics_data[k][0][-1] + 1)
        metrics_data[k][1].append(metrics_data[k][1][-1] * 0.91 + (random.random() - 0.6))

    if clicks % 2 == 0:
        metrics_data['selected'] = [4, 6]

    return metrics_data


@app.callback(
    Output('errors-cache', 'data'),
    [Input('btn-update-errors', 'n_clicks')],
    [State('errors-cache', 'data')],
)
def update_errors_data(clicks, errors_data):
    '''handle updates to the error message data'''
    if clicks is None:
        raise PreventUpdate

    if clicks > 0:
        return mock_error_msgs
    return []


@app.callback(
    Output('errors-list', 'children'),
    [Input('errors-cache', 'data')],
)
def update_errors_list(errors_cache):
    if len(errors_cache) == 0:
        return 'No errors, yay!'
    return render_error_messages(errors_cache)


@app.callback(
    Output('graph_loss', 'figure'),
    [Input('metrics-cache', 'data'), Input('annotations-cache', 'data')],
)
def update_loss(metrics_data, annotations_data):
    graph_figure = {
            'layout': {
                'title': 'Loss over epochs',
            },
            'data': [
                {
                    'x': metrics_data['loss']['train'][0],
                    'y': metrics_data['loss']['train'][1],
                    'name': 'train_200203',
                    'type': 'line+marker',
                },
                {
                    'x': metrics_data['loss']['val'][0],
                    'y': metrics_data['loss']['val'][1],
                    'name': 'val_200203',
                    'type': 'line+marker',
                },
            ],
        }

    if annotations_data:
        annotation_shape = {
            'type': 'rect',
            'xref': 'x',
            'yref': 'paper',
            'x0': annotations_data[0], # x0, x1 are epoch bounds
            'x1': annotations_data[1],
            'y0': 0,
            'y1': 1,
            'fillcolor': 'LightSalmon',
            'opacity': 0.5,
            'layer': 'below',
            'line_width': 0,
        }
        graph_figure['layout']['shapes'] = [annotation_shape]

    return graph_figure


@app.callback(
    Output('graph_acc', 'figure'),
    [Input('metrics-cache', 'data'), Input('annotations-cache', 'data')],
)
def update_loss(metrics_data, annotations_data):
    graph_figure = {
            'layout': {
                'title': 'Accuracy over epochs',
            },
            'data': [
                {
                    'x': metrics_data['acc']['train'][0],
                    'y': metrics_data['acc']['train'][1],
                    'name': 'train_200203',
                    'type': 'line+marker',
                },
                {
                    'x': metrics_data['acc']['val'][0],
                    'y': metrics_data['acc']['val'][1],
                    'name': 'val_200203',
                    'type': 'line+marker',
                },
            ],
        }

    if annotations_data:
        annotation_shape = {
            'type': 'rect',
            'xref': 'x',
            'yref': 'paper',
            'x0': annotations_data[0], # x0, x1 are epoch bounds
            'x1': annotations_data[1],
            'y0': 0,
            'y1': 1,
            'fillcolor': 'LightSalmon',
            'opacity': 0.5,
            'layer': 'below',
            'line_width': 0,
        }
        graph_figure['layout']['shapes'] = [annotation_shape]

    return graph_figure

if __name__ == '__main__':
    app.run_server(debug=True)

