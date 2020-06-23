# -*- coding: utf-8 -*-

import dash_core_components as dcc
import dash_html_components as html

from umserver import app
from umserver.models import get_training_sessions

x = [-1, 0, 1, 2, 3]

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    dcc.Location(id='url-update', refresh=False),
    html.Div([
        html.H1('Umlaut Toolkit', style={'display': 'inline-block'}),
        dcc.Dropdown(
            id='session-picker',
            placeholder='Named Sessions',
            style={'width': 200, 'display': 'inline-block', 'float': 'right'},
        ),
        dcc.Graph(
            id='timeline',
            figure={
                'layout': {
                    'height': 300,
                    'bargap': 0,
                    'yaxis': {
                        'showgrid': False,
                        'zeroline': False,
                        'showline': False,
                        'showticklabels': False,
                        'range': [-1, 1],
                    },
                    'xaxis_title': 'Errors over time',
                },
                'data': [
                    {
                        'x': [0, 1],
                        'y': [0.5, 0.5],
                        'type': 'scatter',
                        'opacity': 0.5,
                        'mode': 'line',
                        'name': 'Overfitting',
                        'line': {'color': 'red', 'width': 20},
                    },
                    {
                        'x': [0, 2],
                        'y': [0, 0],
                        'type': 'scatter',
                        'opacity': 0.5,
                        'mode': 'line',
                        'name': 'Normalization',
                        'line': {'color': 'green', 'width': 20},
                    },
                    {
                        'x': [1, 2],
                        'y': [-0.5, -0.5],
                        'type': 'scatter',
                        'opacity': 0.5,
                        'mode': 'line',
                        'name': 'Plateauing',
                        'line': {'color': 'yellow', 'width': 20},
                    },
                ]
            },
        ),
    ]),
    html.Div([
            html.H3('Visualizations'),
            dcc.Graph(
                id='graph_loss',
                figure={
                    'layout': {'title': 'Loss over Epochs'},
                },
            ),
            dcc.Graph(
                id='graph_acc',
                figure={
                    'layout': {'title': 'Accuracy over Epochs'},
                },
            ),
        ],
        className='five columns',
    ),
    html.Div([
            html.H3('Error Messages', style={'display': 'inline-block'}),
            html.Button(id='btn-clear-annotations', children='Clear Annotations', style={'display': 'inline-block', 'float': 'right'}),
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
