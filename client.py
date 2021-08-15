from __future__ import print_function
import requests
import json
import cv2
import base64

addr = 'http://192.168.0.108:5000/'
test_url = addr + '/api/v1/ai/detection'
content_type = 'image/jpeg'
headers = {'content-type': content_type}
file_name = 'Testing/Input/tung.jpg'
# /api/v1/ai/detection
def post_image_v1(img_file):
    # prepare headers for http request
    img = cv2.imread(img_file)
    # encode image as jpeg
    _, img_encoded = cv2.imencode('.jpg', img)
    # send http request with image and receive response
    response = requests.post(test_url, data=img_encoded.tostring(), headers=headers)
    # decode response
    print(json.loads(response.text))

# /api/v2/ai/detection
def post_image_v2(img_file):
    img = open(img_file, 'rb').read()
    response = requests.post(test_url, data=img, headers=headers)
    print(response.text)

def post_image_v3(img_file):
    img = open(img_file, 'rb').read()
    image_data = base64.b64encode(img).decode()
    response = requests.post(test_url, data=image_data, headers=headers)
    print(response.text)

# post_image_v1(file_name)
post_image_v3(file_name)