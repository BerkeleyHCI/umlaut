from pymongo import MongoClient

client = MongoClient('localhost', 27017)
db = client['umlaut']

__all__ = ['db']
