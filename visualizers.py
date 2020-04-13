#!/usr/bin/env python3
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
    base_url = "http://0.0.0.0:5000/model/api/v1.0/"
    headers = {'Content-Type': 'application/json'}
    image_req = json.dumps({'img': str(encode_img(img))})
    response = requests.request(
        "GET", base_url+'recognize', headers=headers, data=image_req)
    names = json.loads(response.content)['names']
    faces = json.loads(response.content)['faces']
    landmarks = json.loads(response.content)['landmarks']

    # parse predictions
    names = names.replace(r'[', '').replace(r']', '').replace(
        r'"', '').replace(r' ', '').split(',')
    # print('Names :', names)

    faces = faces.replace(r'[', '').replace(r' ', '').split('],')
    faces = [face.replace(r']', '').split(',') for face in faces]
    faces = [[float(pos) for pos in face_pos] for face_pos in faces]
    faces = np.array(faces)
    # print('Faces :', faces)

    landmarks = landmarks.replace(r'[', '').replace(r' ', '').split(']],')
    landmarks = [landmark.split('],') for landmark in landmarks]
    landmarks = [[s.replace(']','').split(',') for s in landmark] for landmark in landmarks]
    landmarks = [[[float(pos) for pos in landmark_pos] for landmark_pos in landmark] for landmark in landmarks]
    landmarks = np.array(landmarks)
    # print('Landmarks :', landmarks)

    for i in range(len(names)):
        box = faces[i].astype(np.int)
        color = (0, 0, 255)
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)
        if landmarks is not None:
            landmark5 = landmarks[i].astype(np.int)
        # print(landmark.shape)
        for l in range(landmark5.shape[0]):
            color = (0, 0, 255)
            if l == 0 or l == 3:
                color = (0, 255, 0)
            cv2.circle(img, (landmark5[l][0], landmark5[l][1]), 1, color, 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (box[2]-80, box[3]+15)
            fontScale = 0.4
            fontColor = (0, 255, 255)
            lineType = 2

            cv2.putText(img, names[i],
                        bottomLeftCornerOfText,
                        font,
                        fontScale,
                        fontColor,
                        lineType)
            cv2.imshow('output', img)

    return img


def visualize_activity(img):
    img[..., 0] = 0
    img[..., 2] = 0
    return img


def visualize_pose(img):
    img[..., :-1] = 0
    return img
