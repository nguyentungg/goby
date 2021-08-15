from flask import Flask, request, Response
from flask_cors import CORS, cross_origin
from train_data import training_model
from utils import Detector
import Core.Helper as help
import jsonpickle
import numpy as np
import cv2
import base64

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
app.config["DEBUG"] = True

UPLOAD_FOLDER = 'received_files'
ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg']


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


@app.route('/api/v1/ai/train', methods=['POST'])
@cross_origin()
def training_model():
    help.crop_image('Dataset/Raw', 'Dataset/Crop')
    training_model('Dataset/Crop', 'DataBase')
    return 'Training Completed!', 200


# route http posts to this method from bytes image data
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
