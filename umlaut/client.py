import requests
from bson import ObjectId
from datetime import datetime as dt
from pymongo import MongoClient

client = MongoClient('localhost', 27017)
db = client['umlaut']

class UmlautClient:
    def __init__(self, session_name=None, host=None):
        # set up host for umlaut server
        self.host = host or 'localhost:8888'
        if not self.host.startswith('http://'):
            self.host = 'http://' + self.host

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
            self.host + f'/api/updateSessionPlots/{self.session_id}',
            json=req_data,
        )
        r.raise_for_status()


    @staticmethod
    def get_sessionid_str_from_name(session_name):
        '''Find a session named session_name, otherwise make it.
        '''
        ses = db.sessions.find_one_and_update(
            {'name': session_name},
            {'$set': {
                'name': session_name,
                'modify_timestamp': dt.isoformat(),
            }},
            upsert=True,
        )
        return str(ses['_id'])  # return string from ObjectId


def update_col_parameters():
    pass