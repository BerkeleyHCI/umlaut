# -*- coding: utf-8 -*-

import dash

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# initialize Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=external_stylesheets,
    suppress_callback_exceptions=True,  # must have this for dynamic callbacks
)

# import API routes
from umserver import api
from umserver import layout
from umserver import callbacks

# expose internal flask object for serving
server = app.server