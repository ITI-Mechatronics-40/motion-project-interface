import requests
import json
import base64
import cv2


base_url = "http://localhost:5000/activity_recognition/i3d/v1.0/init_model"

headers = {
  'Content-Type': 'application/json'
}


def encode_img(image):
    _, buffer = cv2.imencode('.jpg', image)
    enc_buff = base64.b64encode(buffer)
    return str(enc_buff, 'utf-8')


def init_i3d_model(num_samples=16):
    init_req = json.dumps({
        'eval_type': 'rgb',
        'imagenet_pretrained': 'True',
        'image_size': 224,
        'num_of_frames': num_samples
    })
    response = requests.request("POST", base_url, headers=headers, data=init_req)
    return json.loads(response.text.encode('utf8'))['API']


api_url = init_i3d_model(num_samples=16)


def add_to_sample(img):
    global api_url
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(f'Sent Image Shape: {img.shape}')
    image_req = json.dumps({'img': encode_img(img)})
    response = requests.request("PUT", url=api_url['upload_img'], headers=headers, data=image_req)
    data = json.loads(response.content)
    if data['last_prediction'] == '':
        last_prediction = ''
    else:
        last_prediction = data['last_prediction'][0].split(',')
    return response.status_code, last_prediction


def run_activity_inference():
    global api_url
    response = requests.request("GET", api_url['run'])
    return json.loads(response.content)['prediction'][0].split(',')

