# -*- coding: utf-8 -*-

import bson
import re
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


@server.route('/api/getSessionIdFromUniqueName/<sess_name>', methods=['GET'])
def get_session_id_from_making_unique_name(sess_name):
    '''Find a session named session_name, otherwise make it.
    
    If one already exists, add a (safely incremented) _{int} to the end.
    '''
    ses_candidates = list(db.sessions.find({'name': {'$regex': sess_name + '(?:_\d+)?$'}}))
    if not ses_candidates:
        return get_sessionid_str_from_name(sess_name)
    max_incr = 0
    for s in ses_candidates:
        s = s['name']
        incr = re.search('_(\d+)$', s)
        if incr:
            incr = int(incr[1])
            if incr > max_incr:
                max_incr = incr
    sess_name = sess_name + f'_{max_incr + 1}'
    return get_sessionid_str_from_name(sess_name)


@server.route('/api/updateSessionPlots/<sess_id>', methods=['POST'])
def update_session_plots(sess_id):
    '''adds new data from a training session to the db.

    an update is structured as follows:

    updates = {
        'loss': {
            'train': [<int:epochs>, <float:value>],
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
    for plot_name in updates:  # loss, acc
        for plot_col in updates[plot_name]:  # train, val
            update_data = updates[plot_name][plot_col]
            assert len(list(update_data)) == 2  # [epochs, data]
            db.plots.update(
                {'session_id': sess_id, 'name': plot_name},
                {'$push': {'streams.' + plot_col: update_data}},
                upsert=True,
            )
            print(f'epoch {update_data[0]}: {plot_name}.{plot_col} <-+ {update_data[1]}')
    return f'Updated {str(len(updates))}'


@server.route('/api/updateSessionErrors/<sess_id>', methods=['POST'])
def update_session_errors(sess_id):
    '''Receive an error message id and store in the db.'''
    try:
        sess_id = ObjectId(sess_id)
    except bson.errors.InvalidId:
        abort(400)
    if db.sessions.find_one(sess_id) is None:
        abort(404)  # session not found

    errors = request.get_json()
    for error_id in errors:
        error_obj = {
            '$set': {
                'session_id': sess_id,
                'error_id_str': error_id,
            },
        }
        if errors[error_id]['epochs'] is None:
            # don't make a list of None's from global errors, just set once.
            error_obj['$set'].update({'epochs': None})
        else:
            error_obj['$addToSet'] = {
                # $each to iterate through list
                'epochs': {'$each': errors[error_id]['epochs']},
            }
        for k in errors[error_id]:
            if k not in ('epochs', 'session_id', 'error_id_str'):
                # add any remaining keys sent over to the db
                error_obj['$set'].update({k: errors[error_id][k]})

        db.errors.find_one_and_update(
            {'error_id_str': error_id, 'session_id': sess_id},
            error_obj,
            upsert=True,
        )
    return f'Updated {str(len(errors))}'