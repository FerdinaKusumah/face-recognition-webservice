#!/usr/bin/env python3
import os
import pickle
from http import HTTPStatus

from sanic import Sanic
from sanic import response
from sanic.exceptions import NotFound
from sanic_cors import CORS

from handler.routes import services

app = Sanic(__name__)
CORS(app)


app.blueprint(services)


@app.listener('before_server_start')
async def init(app, loop):
    async def load_pickle():
        _path = os.path.dirname(os.path.abspath(__file__)) + '/model/train_model.clf'
        with open(_path, 'rb') as f:
            return pickle.load(f)

    # distance threshold
    app.distance_threshold = 0.4
    # load classifier
    app.train_model = await load_pickle()


@app.listener('after_server_stop')
async def close_connection(app, loop):
    pass


@app.exception(NotFound)
async def ignore_404s(request, exception):
    return response.json({'status': HTTPStatus.NOT_FOUND, 'message': 'Route not found'})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, workers=4)
