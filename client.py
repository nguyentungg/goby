from __future__ import print_function
import requests
import json
import cv2

addr = 'http://localhost:5555'
test_url = addr + '/api/test'

# /api/v1/ai/detection
def post_image_v1():
    # prepare headers for http request
    content_type = 'image/jpeg'
    headers = {'content-type': content_type}

    img = cv2.imread('mona.jpg')
    # encode image as jpeg
    _, img_encoded = cv2.imencode('.jpg', img)
    # send http request with image and receive response
    response = requests.post(test_url, data=img_encoded.tostring(), headers=headers)
    # decode response
    print(json.loads(response.text))

# /api/v2/ai/detection
def post_image_v2(img_file):
    img = open(img_file, 'rb').read()
    response = requests.post(URL, data=img, headers=headers)
    return response
