# -*- coding: utf-8 -*-

import bson
import dash
import json
import random
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from bson import ObjectId
from flask import abort
from flask import request

from umserver import app
from umserver.helpers import argmax
from umserver.models import db

MAX_ERRORS = 10

# ---------- Helper Functions for Rendering ---------- #

def get_go_data_from_metrics(plot, metrics_data):
    '''populate graph_figure.data from metrics_data'''
    return [  # make a plot for every plot stream
        {
            'x': metrics_data[plot][k][0],
            'y': metrics_data[plot][k][1],
            'name': k,
            'type': 'line+marker',
        } for k in metrics_data[plot]
    ]


def make_annotation_box_shape(xbounds):
    return {
        'type': 'rect',
        'xref': 'x',
        'yref': 'paper',
        'x0': xbounds[0], # x0, x1 are epoch bounds
        'x1': xbounds[1],
        'y0': 0,
        'y1': 1,
        'fillcolor': 'LightPink',
        'opacity': 0.5,
        'layer': 'below',
        'line_width': 0,
    }


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
    # don't catch invalidId, let it error out
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
    # args now initialized
    if sum(click_timestamps) == 0:  # application reset or nothing clicked
        raise PreventUpdate
    clicked_idx = argmax(click_timestamps)
    if clicked_idx == len(click_timestamps) - 1:  # pressed the clear annotations button
        return []
    annotations = error_msgs[clicked_idx]['annotations']  #TODO change to object, requires schema change
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
    try:
        session_plots = [s for s in db.plots.find({'session_id': ObjectId(sess_id)})]
    except bson.errors.InvalidId:
        return {}

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

    if interval == 0:
        path = pathname.split('/')
        sess_id = path[path.index('session') + 1]
        try:
            error_msgs = list(db.errors.find(
                {'session_id': ObjectId(sess_id)},
                {'_id': 0, 'session_id': 0},  # omit object ids from results, not json friendly
            ))
        except bson.errors.InvalidId:
            abort(400)
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
        result_divs.append(html.P('No errors yay!'))

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
    if not metrics_data:
        return {}

    graph_figure = {
        'layout': {
            'title': 'Loss over epochs',
        },
        'data': get_go_data_from_metrics('loss', metrics_data),
    }

    if annotations_data:
        graph_figure['layout']['shapes'] = [make_annotation_box_shape(annotations_data)]

    return graph_figure


@app.callback(
    Output('graph_acc', 'figure'),
    [Input('metrics-cache', 'data'), Input('annotations-cache', 'data')],
)
def update_acc(metrics_data, annotations_data):
    if not metrics_data:
        return {}
    graph_figure = {
            'layout': {
                'title': 'Accuracy over epochs',
            },
            'data': get_go_data_from_metrics('acc', metrics_data),
        }

    if annotations_data:
        graph_figure['layout']['shapes'] = [make_annotation_box_shape(annotations_data)]

    return graph_figure

if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8888, debug=True)

