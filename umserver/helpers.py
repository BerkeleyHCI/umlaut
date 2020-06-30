# -*- coding: utf-8 -*-

def argmax(l):
    return max(range(len(l)), key=lambda i: l[i])


def filter_for_dict(l, k, v):
    return next((d for d in l if d[k] == v), None)


def index_of_dict(l, k, v):
    '''return the index of l where l[i] has a dict which has (k, v)'''
    for i, d in enumerate(l):
        if k in d and d[k] == v:
            return i
    return None