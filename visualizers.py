import requests
import json
import base64
import numpy as np
import cv2


def encode_img(image):
    _, buffer = cv2.imencode('.jpg', image)
    enc_buff = base64.b64encode(buffer)
    return str(enc_buff, 'utf-8')


def visualize_faces(img):
	img [...,1:] = 0
	return img


def visualize_activity(img):
	img[...,0] = 0
	img[...,2] = 0
	return img


def visualize_pose(img):
    url = "http://0.0.0.0:5001/"
    headers = {'Content-Type': 'application/json'}
    image_req = json.dumps({'img': str(encode_img(img))})
    response = requests.request("GET", url=url+'analyse_image', headers=headers, data=image_req)
    img = json.loads(response.content)['data']
    img = np.array(img, dtype=np.uint8)

    return img