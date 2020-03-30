# Face recognition webservice using python [sanic](https://github.com/huge-success/sanic)

## Manual install Prerequisites
Needs Python 3.5 +

* Install all dependency
```bash
    $ pip install -r requirements.txt
```

* Run application
```bash
    $ python app.py
```

### This service will display name and face positions from president of Indonesia until 7th President

Routes:
    
    * [GET] - `/api/hello` check if api is live
    * [POST] - `/api/recognize` to recognize an image and return result as base64 image 



## References
* [Sanic framework](https://github.com/huge-success/sanic)
* [Face recognition](https://github.com/ageitgey/face_recognition)
