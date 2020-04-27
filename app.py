# -*- coding: utf-8 -*-

import dash
import json
import random
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from bson import ObjectId

from models import db

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

MAX_ERRORS = 10

app = dash.Dash(
    __name__,
    external_stylesheets=external_stylesheets,
    suppress_callback_exceptions=True,  # must have this for dynamic callbacks
)

def argmax(l):
    return max(range(len(l)), key=lambda i: l[i])


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
            html.Button(id='btn-clear-annotations', n_clicks=0, children='Clear Annotations'),
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
    [Input('error-msg-{}'.format(i), 'n_clicks_timestamp') for i in range(MAX_ERRORS)] + \
            [Input('btn-clear-annotations', 'n_clicks_timestamp'),
             Input('errors-cache', 'data')],
)
def highlight_graph_for_error(*click_timestamps):
    '''change the annotations state based on clicked error msg'''
    len_clicks = len(click_timestamps)
    click_timestamps, error_msgs = click_timestamps[:len_clicks-1], click_timestamps[-1]
    click_timestamps = [i or 0 for i in click_timestamps]
    clicked_idx = argmax(click_timestamps)
    if clicked_idx == len(click_timestamps) - 1 or all(t is None for t in click_timestamps):  # pressed the clear annotations button
        return []
    annotations = error_msgs[clicked_idx]['annotations']
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
        session_data_loss = db.plots.find_one({'session_id': ObjectId('5e8be9a283d409a2560de721'), 'name': 'loss'})
        session_data_acc = db.plots.find_one({'session_id': ObjectId('5e8be9a283d409a2560de721'), 'name': 'acc'})

        #TODO handle case where query is empty
        train_loss = list(zip(*session_data_loss['train']))
        val_loss = list(zip(*session_data_loss['val']))
        train_acc = list(zip(*session_data_acc['train']))
        val_acc = list(zip(*session_data_acc['val']))
        data = {  # for now, keep the same format as before... we don't have to
            'loss': {
                'train': train_loss,
                'val': val_loss,
            },
            'acc': {
                'train': train_acc,
                'val': val_acc,
            },
        }
        return data


    for k in metrics_data['loss'].keys():  #TODO remove, only to simulate new data points
        metrics_data['loss'][k][0].append(metrics_data['loss'][k][0][-1] + 1)
        metrics_data['loss'][k][1].append(max(metrics_data['loss'][k][1][-1] * 0.91 + (random.random() - 0.6), 0))

        metrics_data['acc'][k][0].append(metrics_data['acc'][k][0][-1] + 1)
        metrics_data['acc'][k][1].append(min(metrics_data['acc'][k][1][-1] * 1.01 + (random.random() * 2), 100))

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

    #TODO make this actually depend on errors_data
    if clicks == 0:
        error_msgs = list(db.errors.find(
            {'session_id': ObjectId('5e8be9a283d409a2560de721')},
            {'_id': 0, 'session_id': 0},  # omit object ids from results, not json friendly
        ))
        return error_msgs

    return errors_data


@app.callback(
    Output('errors-list', 'children'),
    [Input('errors-cache', 'data')],
)
def update_errors_list(errors_data):
    if len(errors_data) == 0:
        return 'No errors, yay!'

    result_divs = []
    for i, error in enumerate(errors_data):
        result_divs.append(render_error_message('error-msg-{}'.format(i), error))

    for i in range(len(result_divs), MAX_ERRORS):
        result_divs.append(html.Div(id='error-msg-{}'.format(i), style={'display': 'none'}))

    return result_divs


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
    app.run_server(host='0.0.0.0', port=8888, debug=True)

