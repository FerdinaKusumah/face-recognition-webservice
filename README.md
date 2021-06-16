# Face recognition webservice using python [sanic](https://github.com/huge-success/sanic)

## Manual install Prerequisites
Needs Python 3.5 +

* Install all dependency
```bash
    $ pip install -r requirements.txt
```

* Run application
```bash
    $ python main.py
```

### This service will display name and face positions from famous technology founder

## Train process
* Go to train folder then run  `python3 train.py` to train all images inside train folder then save all training result to folder model in root dir
    * Note: this may take sometime to wait train all images to finish.
* Or you can use prebuild model inside folder model instead.


# Testing api
Routes:

    * [GET] - `/api/hello` check if api is live
    * [POST] - `/api/recognize` to recognize an image and return result as base64 image


# Result

![Image 1](train/test/example_1.png)

![Image 2](train/test/example_2.png)



## References
* [Sanic framework](https://github.com/huge-success/sanic)
* [Face recognition](https://github.com/ageitgey/face_recognition)
