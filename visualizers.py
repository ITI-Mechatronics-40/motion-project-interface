#!/usr/bin/env python3
import cv2
import numpy as np

from activity_service import add_to_sample, run_activity_inference

import requests
import json
import base64

import configparser

config = configparser.ConfigParser()
config.read("config.ini")

def encode_img(image):
    _, buffer = cv2.imencode('.jpg', image)
    enc_buff = base64.b64encode(buffer)
    return str(enc_buff, 'utf-8')

def visualize_text(img, text):
	font = cv2.FONT_HERSHEY_SIMPLEX
	font_scale = 1
	thickness = 1
	line_height = np.ceil(0.075 * img.shape[0])
	line_width = np.ceil(0.35 * img.shape[1])
	line_spacing = np.ceil(0.025 * img.shape[0])
	for idx, line in enumerate(text):
		((s_x, s_y), _) = cv2.getTextSize(line, font,font_scale, thickness)
		if s_x > line_width:
			font_scale_x = int((line_width / s_x) / 0.1) * 0.1
			s_x *= font_scale_x
			s_y *= font_scale_x
			font_scale *= font_scale_x
		if s_y > line_height:
			font_scale_y = int((line_height / s_y) / 0.1) * 0.1
			s_x *= font_scale_y
			s_y *= font_scale_y
			font_scale *= font_scale_y
		cv2.putText(img, line, (10,int((idx+1) * (line_spacing + s_y))), font, font_scale, (0, 0, 255), thickness, cv2.LINE_AA)
	return img


def visualize_activity(api_url, img):
	status_code, last_prediction = add_to_sample(api_url, img)
	text = ['', '']
	if status_code == 200:
		prediction = run_activity_inference(api_url)
		text = [f'Label: {prediction[2]}', f'Confidence: {float(prediction[1]) * 100 :0.3f}%']
	elif status_code == 202:
		if last_prediction != '':
			text = [f'Label: {last_prediction[2]}', f'Confidence: {float(last_prediction[1]) * 100 :0.3f}%']
	img = visualize_text(img, text)
	return img
	
	
def visualize_faces(img):
    base_url = "http://0.0.0.0:5000/model/api/v1.0/"
    headers = {'Content-Type': 'application/json'}
    image_req = json.dumps({'img': str(encode_img(img))})
    response = requests.request(
        "POST", base_url+'recognize', headers=headers, data=image_req)
    try:
        names = json.loads(response.content)['names']
        faces = json.loads(response.content)['faces']
        landmarks = json.loads(response.content)['landmarks']

        # parse predictions
        names = names.replace(r'[', '').replace(r']', '').replace(r'"', '').replace(r' ', '').split(',')
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
                # print(landmark5.shape)
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
            # cv2.imshow('output', img)
    except:
        pass

    return img

def visualize_pose(img):
    url = "http://{}/".format(config['HOSTNAMES']['pose_service'])
    headers = {'Content-Type': 'application/json'}
    image_req = json.dumps({'img': str(encode_img(img))})
    response = requests.request("GET", url=url+'analyse_image', headers=headers, data=image_req)
    img = json.loads(response.content)['data']
    img = np.array(img, dtype=np.uint8)
    return img

