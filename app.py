#!/usr/bin/env python3
from sanic import Sanic
from sanic import response
from sanic.exceptions import NotFound
import pickle as cPickle

from handler.services import services

app = Sanic(__name__)
app.blueprint(services)
app.config['model_path'] = './model/train_model.clf'

@app.listener('before_server_start')
async def init(sanic, loop):
    with open(app.config['model_path'], 'rb') as train_model:
        knn_classifier = cPickle.load(train_model)
        app.config['db'] = knn_classifier

@app.listener('after_server_stop')
async def close_connection(app, loop):
    pass

@app.exception(NotFound)
async def ignore_404s(request, exception):
    return response.json({'status': 500, 'message': 'Route not found'})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=19999, debug=True, workers=4)