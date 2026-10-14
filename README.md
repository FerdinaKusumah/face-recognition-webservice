# Face recognition webservice using python [sanic](https://github.com/huge-success/sanic)

> **⚠️ Archived: this project is no longer maintained.**
>
> It was written as an example in 2018 and its dependencies are pinned to old versions: Python 3.8 (end of life since
> October 2024), Pillow 9.0.1 (newer releases contain security fixes) and Sanic 21.12. Don't deploy it as it is.
> For new projects, look at maintained face recognition libraries such as
> [DeepFace](https://github.com/serengil/deepface) or [InsightFace](https://github.com/deepinsight/insightface).

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
