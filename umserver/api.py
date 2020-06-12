# -*- coding: utf-8 -*-

import bson
from bson import ObjectId
from datetime import datetime as dt
from flask import abort
from flask import request
from pymongo import ReturnDocument

from umserver import app
from umserver.models import db

# get the internal flask object for client facing API
server = app.server


@server.route('/api/getSessionIdFromName/<sess_name>', methods=['GET'])
def get_sessionid_str_from_name(sess_name):
    '''Find a session named session_name, otherwise make it.
    '''
    ses = db.sessions.find_one_and_update(
        {'name': sess_name},
        {'$set': {
            'name': sess_name,
            'modify_timestamp': dt.now().isoformat(),
        }},
        upsert=True,
        return_document=ReturnDocument.AFTER,
    )
    return str(ses['_id'])  # return string from ObjectId


@server.route('/api/updateSessionPlots/<sess_id>', methods=['POST'])
def update_session_plots(sess_id):
    '''adds new data from a training session to the db.

    an update is structured as follows:

    updates = {
        'loss': {
            'train': [<int:epoch>, <float:value>],
            ...,
        },
        ...,
    }
    '''
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
                {'$push': {'streams.' + plot_col: update_data}},
                upsert=True,
            )
            print(f'epoch {update_data[0]}: {plot_name}.{plot_col} <-+ {update_data[1]}')
    return 'done!'