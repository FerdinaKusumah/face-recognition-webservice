import face_recognition
import json as JSON
import math, uuid, re, time, io
import pickle as cPickle

from sanic import Sanic, Blueprint, response
from sklearn import neighbors

from helper import beautify

services = Blueprint('services')

# call this api to make sure api is running
@services.get('/api/hello', strict_slashes=True)
async def hello(request):
    return response.json({'status': 200, 'message': 'Hello .. '})

# function to validate acceptable files
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# function to predict unknown image
# this library use KNN model 
async def predict(stream_image, knn_clf, distance_threshold=0.6):

    # Load image file and find face locations
    X_img = face_recognition.load_image_file(io.BytesIO(stream_image))
    X_face_locations = face_recognition.face_locations(X_img)

    # If no faces are found in the image, return an empty result.
    if len(X_face_locations) == 0:
        return []

    # Find encodings for faces in the test iamge
    faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)

    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

    # Predict classes and remove classifications that aren't within the threshold
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]

# get prediction name based on unknown image
async def get_prediction_names(predictions):
    retval = []
    for name, (top, right, bottom, left) in predictions:
        retval.append({'name': name, 'face_locations': {'top': top, 'left': left}})
    
    return retval

# load pickle model
async def get_pickle_model(request):
    return request.app.config['db']

# api that handle recognition image
@services.post('/api/face-recognition/recognize-image', strict_slashes=True)
async def detect(request):
    if 'file' not in request.files:
        return response.json({'status': 500, 'message': 'image is required'})
    
    file = request.files.get('file')
    if file.name == '':
        return response.json({'status': 500, 'message': 'image is required name'})
    
    if not file and not allowed_file(file.name):
        return response.json({'status': 500, 'message': 'image extension not allowed'})

    # get file stream
    file_stream = file.body
    knn_clf_data = await get_pickle_model(request)

    # train data from db to Knn
    predictions = await predict(file_stream, knn_clf_data, distance_threshold=0.6)
    results = await get_prediction_names(predictions)
    return response.json({'status': 200, 'data': results})