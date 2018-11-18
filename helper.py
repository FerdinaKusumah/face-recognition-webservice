from sanic import Sanic, Blueprint, response

helper = Blueprint('helper')

async def beautify(dataCursor):
    for doc in dataCursor:
        doc['id'] = str(doc['_id'])
        del doc['_id']
    
    return dataCursor