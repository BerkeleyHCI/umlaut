import requests
from bson import ObjectId
from datetime import datetime as dt
from pymongo import MongoClient
from pymongo import ReturnDocument


class UmlautClient:
    def __init__(self, session_name=None, host=None, port=5000):
        # set up host for umlaut server
        self.host = host or 'localhost'
        self.host = host + f':{port}'

        #TODO remove when replacing db calls with API calls
        client = MongoClient(host, 27017)
        self.db = client['umlaut']

        # get session id from database, whether existing or new
        if not session_name:
            # if no name, unnamed_{yymmdd_hhmmss} is used
            session_name = 'unnamed_' + dt.strftime(dt.now(), '%y%m%d_%H%M%S')
        self.session_id = self.get_session_id_from_name(session_name)


    def get_session_id_from_name(self, session_name):
        r = requests.get(
            f'http://{self.host}/api/getSessionIdFromName/{session_name}',
        )
        r.raise_for_status()
        return r.content


    def send_logs_to_server(self, batch, logs):
        if not logs:
            return
        val = any(k.startswith('val') for k in logs)
        acc = any(k.startswith('acc') for k in logs)
        metrics_dict = {
            'loss': {
                'train': [batch, float(logs['loss'])],
            },
        }
        if val:
            metrics_dict['loss']['val'] = [batch, float(logs['val_loss'])]
        if acc:
            metrics_dict['acc'] = {
                'train': [batch, float(logs['accuracy'] if self.tf_version == 2 else logs['acc'])],
            }
            if val:
                metrics_dict['acc']['val'] = [batch, float(logs['val_accuracy'] if self.tf_version == 2 else logs['val_acc'])]
        self._send_batch_metrics(metrics_dict)


    def _send_batch_metrics(self, req_data):
        '''send updated metrics to umlaut server.
        looks like:
        {
            'acc': {'train': [6, 72.1], 'val': [6, 47.8]},
            'loss': {'train': [6, 0.42], 'val': [6, 0.84]},
        }
        '''
        r = requests.post(
            f'http://{self.host}/api/updateSessionPlots/{self.session_id}',
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
