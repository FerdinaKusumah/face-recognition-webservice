from http import HTTPStatus

from sanic import Blueprint, response
from handler.services import Prediction, Helper

services = Blueprint('services')


# call this api to make sure api is running
@services.get('/api/hello', strict_slashes=True)
async def hello(request):
    return response.json({'status': HTTPStatus.OK, 'message': 'Hello .. '})


# api that handle recognition image
@services.post('/api/recognize', strict_slashes=True)
async def recognize(request):
    if 'image' not in request.files:
        return response.json({'status': HTTPStatus.BAD_REQUEST, 'message': 'image is required'})

    file = request.files.get('image')
    if file.name == '':
        return response.json({'status': HTTPStatus.BAD_REQUEST, 'message': 'image is required name'})

    if not file and not Helper.allowed_file(file.name):
        return response.json({'status': HTTPStatus.BAD_REQUEST, 'message': 'image extension not allowed'})

    train_model = request.app.train_model
    distance_threshold = request.app.distance_threshold
    prediction = Prediction(train_model, distance_threshold)
    # get file extension
    image_extension = file.name.split('.')[1]
    # get file stream
    file_stream = file.body
    results = prediction.predict_image(file_stream, image_extension)
    return response.json({'status': HTTPStatus.OK, 'data': results})
