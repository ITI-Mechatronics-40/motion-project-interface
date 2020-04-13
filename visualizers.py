import requests
import json
import base64
import numpy as np
import cv2


last_prediction = []


def encode_img(image):
    _, buffer = cv2.imencode('.jpg', image)
    enc_buff = base64.b64encode(buffer)
    return str(enc_buff, 'utf-8')


def visualize_faces(img):
    img[..., 1:] = 0
    return img


def visualize_activity(img, api):
    global last_prediction
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image_req = json.dumps({'img': encode_img(rgb_img)})
    headers =   {
                'Content-Type': 'application/json'
                }
    response = requests.request("PUT", url=api['upload_img'], headers=headers, data=image_req)
    shape = img.shape
    if response.status_code == 200:
        response = requests.request("GET", url=api['run'])
        last_prediction = json.loads(response.content)['prediction'][0].strip().split(',')
    elif response.status_code == 400:
        raise AttributeError('Image Size doesn\'t match previous images')
    elif response.status_code == 503:
        raise ConnectionAbortedError('Internal Server Error')
    if last_prediction:
        label = last_prediction[2]
        confidence = float(last_prediction[1])
        label_pos = (5, np.int(shape[0] * 0.9))
        conf_pos = (5, np.int(shape[0] * 0.95))
        img = cv2.putText(img, f'Label: {label}', label_pos, cv2.FONT_HERSHEY_SIMPLEX, 1,
                          (0, 0, 255), 1, cv2.LINE_AA)
        img = cv2.putText(img, f'Confidence:{confidence:.3f}', conf_pos, cv2.FONT_HERSHEY_SIMPLEX, 1,
                          (0, 0, 255), 1, cv2.LINE_AA)
    else:
        label_pos = (5, np.int(shape[0] * 0.95))
        img = cv2.putText(img, 'Loading...', label_pos, cv2.FONT_HERSHEY_SIMPLEX, 1,
                          (0, 0, 255), 1, cv2.LINE_AA)
    return img


def visualize_pose(img):
    img[..., :-1] = 0
    return img
