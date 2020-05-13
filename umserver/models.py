from pymongo import MongoClient

client = MongoClient('localhost', 27017)
db = client['umlaut']


def get_training_sessions():
    '''query mongodb for all sessions'''
    sessions = []
    for sess in db.sessions.find():
        sessions.append({
            'label': sess['name'],
            'value': str(sess['_id']),
        })
    return sessions


__all__ = ['db']
