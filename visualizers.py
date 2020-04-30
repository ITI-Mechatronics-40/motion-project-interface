import cv2
import numpy as np

from activity_service import add_to_sample, run_activity_inference


def visualize_faces(img):
	img [...,1:] = 0
	return img


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
		cv2.putText(img, line, (10,int((idx+1) * (line_spacing + s_y))), font, font_scale,
														(0, 0, 255), thickness, cv2.LINE_AA)
	return img


def visualize_activity(img):
	status_code, last_prediction = add_to_sample(img)
	text = ['', '']
	if status_code == 200:
		prediction = run_activity_inference()
		text = [f'Label: {prediction[2]}', f'Confidence: {float(prediction[1]) * 100 :0.3f}%']
	elif status_code == 202:
		if last_prediction != '':
			text = [f'Label: {last_prediction[2]}', f'Confidence: {float(last_prediction[1]) * 100 :0.3f}%']
	img = visualize_text(img, text)
	return img


def visualize_pose(img):
	img[...,:-1] = 0
	return img