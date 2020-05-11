# -*- coding: utf-8 -*-

import dash
import json
import random
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from bson import ObjectId
from flask import request

from models import db

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

MAX_ERRORS = 10

app = dash.Dash(
    __name__,
    external_stylesheets=external_stylesheets,
    suppress_callback_exceptions=True,  # must have this for dynamic callbacks
)

# ----------- Flask API routes ---------- #

# get the internal flask object for client facing API
server = app.server

@server.route('/api/updateSessionPlots/<sess_id>', methods=['POST'])
def update_session_plots(sess_id):
    try:
        sess_id = ObjectId(sess_id)
    except bson.errors.InvalidId:
        abort(400)
    if db.sessions.find_one(sess_id) is None:
        abort(404)  # session not found

    updates = request.get_json()
    update_plots = updates.keys()
    for plot_name in update_plots:  # loss, acc
        for plot_col in updates[plot_name]:  # train, val
            update_data = updates[plot_name][plot_col]
            assert len(list(update_data)) == 2  # [epoch, data]
            db.plots.update(
                {'session_id': sess_id, 'name': plot_name},
                {'$push': {plot_col: update_data}},
            )
            print(f'epoch {update_data[0]}: {plot_name}.{plot_col} <-+ {update_data[1]}')
    return 'done!'


# ----------- Helpers ---------- #

def argmax(l):
    return max(range(len(l)), key=lambda i: l[i])


def get_training_sessions():
    sessions = []
    for sess in db.sessions.find():
        sessions.append({
            'label': sess['name'],
            'value': str(sess['_id']),
        })
    return sessions


# ----------- App Layout ---------- #

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    dcc.Location(id='url-update', refresh=False),
    html.Div([
        html.H1('Umlaut Toolkit'),
        dcc.Dropdown(
            id='session-picker',
            options=get_training_sessions(),
        ),
        html.Hr(),
    ]),
    html.Div([
            html.H3('Visualizations'),
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
            html.H3('Error Messages'),
            html.Hr(),
            html.Div(id='errors-list'),
        ],
        className='six columns',
    ),
    dcc.Interval(
	id='interval-component',
	interval=10*1000, # in milliseconds
	n_intervals=0,
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
    ], id=id, style={'cursor': 'pointer', 'display': 'inline-block'}, key=id)


# ---------- App Callbacks ---------- #

@app.callback(
    Output('session-picker', 'value'),
    [Input('url-update', 'pathname')],
)
def update_dropdown_from_url(pathname):
    '''set the dropdown value from a different url object

    Yes I know this is awful, but it works. This will make
    pages load from URL alone, populating the dropdown with
    a default value.
    '''
    if pathname and 'session' in pathname:
        path = pathname.split('/')
        sess_id = path[path.index('session') + 1]  # fetch session from URL: /session/session_id
        return sess_id
    return pathname

@app.callback(
    Output('url', 'pathname'),
    [Input('session-picker', 'value')]
)
def change_url(ses):
    '''update URL based on session picked in dropdown'''
    if ses is '/':
        raise PreventUpdate
    return f'/session/{ObjectId(ses)}'


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
    [Input('interval-component', 'n_intervals'), Input('url', 'pathname')],
    [State('metrics-cache', 'data')],
)
def update_metrics_data(intervals, pathname, metrics_data):
    '''handle updates to the metrics data'''
    if intervals is None or pathname is None:
        raise PreventUpdate

    if pathname == '/':
        return {}
    path = pathname.split('/')
    sess_id = path[path.index('session') + 1]  # fetch session from URL: /session/session_id
    session_plots = [s for s in db.plots.find({'session_id': ObjectId(sess_id)})]

    go_data = {}
    for plot in session_plots:
        # this one liner unzips each stream from [[x, y], ...] to [[x, ...], [y, ...]]
        go_data[plot['name']] = {k: list(zip(*plot['streams'][k])) for k in plot['streams']}

    return go_data


@app.callback(
    Output('errors-cache', 'data'),
    [Input('interval-component', 'n_intervals'), Input('url', 'pathname')],
    [State('errors-cache', 'data')],
)
def update_errors_data(interval, pathname, errors_data):
    '''handle updates to the error message data'''
    if interval is None or pathname is None:
        raise PreventUpdate

    #TODO make this actually depend on errors_data
    if interval == 0:
        path = pathname.split('/')
        sess_id = path[path.index('session') + 1]

        error_msgs = list(db.errors.find(
            {'session_id': ObjectId(sess_id)},
            {'_id': 0, 'session_id': 0},  # omit object ids from results, not json friendly
        ))
        return error_msgs

    return errors_data


@app.callback(
    Output('errors-list', 'children'),
    [Input('errors-cache', 'data')],
)
def update_errors_list(errors_data):
    result_divs = []
    
    if not errors_data:
        return html.P('No errors found for this session.')

    if len(errors_data) == 0:
        result_divs.append(html.p('No errors yay!'))

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
    #TODO fix situation where query is empty
    if not metrics_data:
        return {}

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

