import requests
from bson import ObjectId
from datetime import datetime as dt
from pymongo import MongoClient
from pymongo import ReturnDocument


class UmlautClient:
    def __init__(self, session_name=None, host=None):
        # set up host for umlaut server
        self.host = host or 'localhost'

        client = MongoClient(host, 27017)
        self.db = client['umlaut']

        # get session id from database, whether existing or new
        if not session_name:
            # if no name, unnamed_{yymmdd_hhmmss} is used
            session_name = 'unnamed_' + dt.strftime(dt.now(), '%y%m%d_%H%M%S')
        self.session_id = self.get_sessionid_str_from_name(session_name)


    def send_batch_metrics(self, req_data):
        '''send updated metrics to umlaut server.
        looks like:
        {
            'acc': {'train': [6, 72.1], 'val': [6, 47.8]},
            'loss': {'train': [6, 0.42], 'val': [6, 0.84]},
        }
        '''
        r = requests.post(
            f'http://{self.host}:5000/api/updateSessionPlots/{self.session_id}',
            json=req_data,
        )
        r.raise_for_status()


    #TODO make this an API call instead
    def send_errors(self, errors):
        '''send and format an error message (for now)'''
        for error in filter(None, errors):
            self.db.errors.find_one_and_update(
                {'error_id_str': error.id_str, 'epoch': error.epoch},
                {'$set': {
                    'session_id': ObjectId(self.session_id),
                    'error_id_str': error.id_str,
                    'epoch': error.epoch,
                    'annotations': error.annotations,
                    'title': error.title,
                    'description': error.description,
                }},
                upsert=True,
            )


    #TODO make this an API call instead
    def get_sessionid_str_from_name(self, session_name):
        '''Find a session named session_name, otherwise make it.
        '''
        ses = self.db.sessions.find_one_and_update(
            {'name': session_name},
            {'$set': {
                'name': session_name,
                'modify_timestamp': dt.now().isoformat(),
            }},
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )
        return str(ses['_id'])  # return string from ObjectId
