'''example of data API to feed metrics into mongodb

format looks like this:

    'plot_name': {
        'plot_col': {
            [epoch, raw_value],
            ...,
        },
        ...,
    }
'''
import requests


r = requests.post(
    'http://localhost:8888/api/updateSessionPlots/5e8be9a283d409a2560de721',
    json={
        'acc': {'train': [6, 72.1], 'val': [6, 47.8]},
        'loss': {'train': [6, 0.42], 'val': [6, 0.84]},
    },
)
