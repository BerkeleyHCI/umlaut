# -*- coding: utf-8 -*-

import dash_core_components as dcc
import dash_html_components as html

from umlaut import app
from umlaut.models import get_training_sessions

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
