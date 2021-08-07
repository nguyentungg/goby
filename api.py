import flask
from train_data import training_model
import Core.Helper as help
from utils import Detector
from flask import request, jsonify, redirect
from flask_cors import CORS, cross_origin

app = flask.Flask(__name__)
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
    help.crop_image('Dataset/Raw','Dataset/Crop')
    training_model('Dataset/Crop', 'DataBase')
    return 'Training Completed!', 200


@app.route('/api/v1/ai/detection', methods=['POST'])
@cross_origin()
def detect_face():
    if request.method == 'POST':
        detector = Detector()
        img = request.url #Read real image not image path
        img_pre = help.image_preprocessing(img)
        predictions = detector.get_people_names(img_pre, speed_up=False, downscale_by=1)
        return jsonify(predictions)


if __name__ == '__main__':
    app.run(host='0.0.0.0')
