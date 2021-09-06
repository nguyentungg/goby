import os
from flask import Flask, request, Response
from flask_cors import CORS, cross_origin
import train_data as td
from utils import Detector
import Core.Helper as help
import jsonpickle
import numpy as np
import cv2
import base64
import re
import sys

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
app.config["DEBUG"] = True

UPLOAD_FOLDER = 'received_files'
ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg']
# create detector object
if not td.check_database('./DataBase/DataBase.json'):
    detector = Detector(False)
    detector.load_detect()
else:
    detector = Detector()


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET'])
@cross_origin()
def encode_images():
    return '''<h1>Kintai AI</h1>
    <h2>Market Enterprise Vietnam AI Laboratory</h2>
<p>List API</p><p>/api/v1/ai/train</p><p>/api/v1/ai/detection</p>'''


@app.errorhandler(404)
@cross_origin()
def page_not_found(e):
    return "<h1>Kintai AI</h1><h1>404</h1><p>The resource could not be found.</p>", 404


@app.route('/api/v1/ai/crop-image', methods=['POST'])
@cross_origin()
def crop():
    help.crop_images('./Dataset/Raw', './Dataset/Crop', detector)
    return 'Face Crop completed!', 200


@app.route('/api/v1/ai/train', methods=['POST'])
@cross_origin()
def train():
    td.training_model('./Dataset/Crop')
    return 'Training Completed!', 200

# Crop face image and training data
@app.route('/api/v1/ai/crop-train', methods=['POST'])
@cross_origin()
def crop_train():
    help.crop_images('./Dataset/Raw', './Dataset/Crop', detector)
    td.training_model('./Dataset/Crop')
    return 'Training Completed!', 200


@app.route('/api/v1/ai/detection', methods=['POST'])
@cross_origin()
def detection():
    if request.method != 'POST':
        return 'Accept only POST method'
    if detector.recog_isload is False:
        if os.path.isfile('./DataBase/DataBase.json'):
            detector.load_recog()
            print("FaceRecog is loaded", file=sys.stderr)
        else:
            print("There is no Database model", file=sys.stderr)
            return 'Please import the Database model'

    r = request
    image_data = re.sub('^data:image/.+;base64,', '', r.form['photo'])
    jpg_original = base64.b64decode(image_data)
    jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
    # decode image
    img = cv2.imdecode(jpg_as_np, flags=1)
    # convert image color to COLOR_BGR2RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # detect face
    response = detector.get_people_only_names(img, speed_up=False, downscale_by=1)
    response_pickled = jsonpickle.encode(response)
    # print(response_pickled, file=sys.stderr)
    return Response(response=response_pickled, status=200, mimetype="application/json")


# ------------------------- Test function -------------------------
# Route http posts to this method from bytes image data
@app.route('/api/v1/ai/detection_v1', methods=['POST'])
@cross_origin()
def detect_face_v1():
    if request.method != 'POST':
        return 'Accept only POST method'
    r = request
    # convert string of image data to uint8
    nparr = np.fromstring(r.data, np.uint8)
    # decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # convert image color to COLOR_BGR2RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # create detector object
    detector = Detector()
    # # detect face
    response = detector.get_people_only_names(img, speed_up=False, downscale_by=1)

    # encode response using jsonpickle
    response_pickled = jsonpickle.encode(response)

    return Response(response=response_pickled, status=200, mimetype="application/json")


# route http posts to this method from base64 string image data
@app.route('/api/v1/ai/detection_v2', methods=['POST'])
@cross_origin()
def detect_face_v2():
    if request.method != 'POST':
        return 'Accept only POST method'
    r = request
    # convert string of image data to uint8
    jpg_original = base64.b64decode(r.data)
    jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
    # decode image
    img = cv2.imdecode(jpg_as_np, flags=1)
    # convert image color to COLOR_BGR2RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # create detector object
    detector = Detector()
    # # detect face
    response = detector.get_people_only_names(img, speed_up=False, downscale_by=1)
    # encode response using jsonpickle
    response_pickled = jsonpickle.encode(response)

    return Response(response=response_pickled, status=200, mimetype="application/json")


if __name__ == '__main__':
    app.run(host='0.0.0.0')
