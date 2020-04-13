#!/usr/bin/env python
"""
Extracts faces from images and recognize the persons inside each image 
and returns the images the bounding boxes and the recognized faces
"""
# MIT License
#
# Copyright (c) 2020 Moetaz Mohamed
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import numpy as np
import tensorflow as tf
import cv2
import base64
import json
from collections import Counter
from flask import Flask, request, redirect, jsonify, url_for, abort, make_response
from facenet.src import facenet
from insightface.RetinaFace.retinaface import RetinaFace

# Configuration parameters
detection_model = 'retinaface-R50/R50'                     # Name of the detetion model for example 'R50' for LResNet50E
det_epoch = 0                                              # Detection model epoch number
recognition_model = '20180402-114759/'                     # Name of the folder containing the recognition model inside the specified models folder


# Don't change the next set of parameters unless necessary
det_model = 'models/detection/' + detection_model    		# path to the detection model
det_threshold = 0.8
image_size = 160                                       		# check recognition model input layer before changing this value
margin = 20                                             	# Number of margin pixels to crop faces function
gpuid = -1						 	# use CPU
rec_model_folder = 'models/recognition/'+ recognition_model  	# path to the folder contating the recognition model
dataset_binary = 'data/features_data.npy'               	# path to the file containing the recognition dataset features
labels_binary = 'data/labels.npy'                       	# path to the file containing the recognition dataset labels

app = Flask(__name__)
detector = RetinaFace(det_model, det_epoch, gpuid, 'net3')

def decode_img(img_str):
    img_bytes = bytes(img_str, 'utf-8')
    img_buff = base64.b64decode(img_bytes)
    img_jpg = np.frombuffer(img_buff, dtype=np.uint8)
    img = cv2.imdecode(img_jpg, cv2.IMREAD_COLOR)
    return img

def dataset_add(image, label):
    if image.ndim != 2 and image.ndim != 3:
        print('expected input image dimension to be 2 or 3 but got data with {}'.format(image.ndim))
        abort(412)

    face, _ = detect_faces(image)
    if face.shape[0] != 1:
        print('expected image with number of faces = 1 but the detector detected {}'.format(face.shape[0]))
        abort(412)
    
    face = align_faces(image, face)
    face_emb = extract_features(face) 
    emb_array = []
    labels = []
    try:
        emb_array = np.load(dataset_binary)
        labels = np.load(labels_binary)
    except:
        print('error reading recognition dataset features one of the files not found or is corrupted')
        emb_array = np.asarray([])
        labels = np.asarray([])
    
    np.append(emb_array, face_emb)
    np.append(labels, label)
    
    np.save(dataset_binary, emb_array)
    np.save(labels_binary, labels)
    return True
    
def recognition_handle(image):
    if image.ndim != 2 and image.ndim != 3:
        print('expected input image dimension to be 2 or 3 but got data with {}'.format(image.ndim))
        abort(412)
        
    faces_bb, landmarks = detect_faces(image)
    if faces_bb.shape[0] == 0:
        print('No Faces found in the image')
        abort(412)
    faces = align_faces(image, faces_bb)
    faces_emb = extract_features(faces)
    emb_array = []
    labels = []
    try:
        emb_array = np.load(dataset_binary)
        labels = np.load(labels_binary)
    except:
        print('error reading recognition dataset features one of the files not found or is corrupted')
        abort(412)
    print(faces.shape)
    print(landmarks.shape)
    return [KNN_predict(faces_emb[i], emb_array, labels, 3) for i in range(faces.shape[0])] , faces_bb, landmarks
     
    
                      
def detect_faces(frame):
    if frame.ndim == 2:
        frame = facenet.to_rgb(frame)
    im_shape = frame.shape
    scales = [1024, 1980]
    target_size = scales[0]
    max_size = scales[1]
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    scales = [im_scale]
    flip = False
    return detector.detect(frame, threshold=det_threshold, scales=scales, do_flip=flip)

def align_faces(frame, faces_loc):
    nrof_faces = faces_loc.shape[0]
    faces = np.zeros((nrof_faces, image_size, image_size, 3))
    if nrof_faces > 0:
        det = faces_loc[:, 0:4]
        det_arr = []
        img_size = np.asarray(frame.shape)[0:2]
        for i in range(nrof_faces):
            det_arr.append(np.squeeze(det[i]))

        for i, det in enumerate(det_arr):
            det = np.squeeze(det)
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0]-margin/2, 0)
            bb[1] = np.maximum(det[1]-margin/2, 0)
            bb[2] = np.minimum(det[2]+margin/2, img_size[1])
            bb[3] = np.minimum(det[3]+margin/2, img_size[0])
            cropped = frame[bb[1]:bb[3], bb[0]:bb[2], :]
            faces[i] = cv2.resize(cropped, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    return faces

def extract_features(faces):

    with tf.Graph().as_default():
        with tf.Session() as sess:
            facenet.load_model(rec_model_folder)
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            # embedding_size = embeddings.get_shape()[1]
            
            if len(faces.shape) == 3:
                faces = faces[None, :, :, :]
                    
            for i in range(faces.shape[0]):
                faces[i] = cv2.resize(faces[i],(image_size, image_size))
                faces[i] = facenet.prewhiten(faces[i])
                # faces[i] = facenet.crop(faces[i], False, image_size)
                faces[i] = facenet.flip(faces[i], False)
            feed_dict = { images_placeholder:faces, phase_train_placeholder:False }
            return sess.run(embeddings, feed_dict=feed_dict)
                
            
            
def KNN_predict(image_embeddings, data_embeddings, labels, k):
    distances = []
    for i in range(len(data_embeddings)):
        distances.append((facenet.distance(image_embeddings, data_embeddings[i], distance_metric = 0),labels[i]))
    distances = sorted(distances, key=lambda tup: tup[0])
    max_iters = (Counter(elem[1] for elem in distances[:min(k, len(data_embeddings))]))
    result = ''
    curr_freq = 0
    for key, occur in max_iters.items():
        if occur > curr_freq:
            curr_freq = occur
            result = key
    return result


@app.route('/model/api/v1.0/recognize', methods=['GET'])
def recognize_image():
    if not request.json or not 'img' in request.json:
       abort(204)
    img = decode_img(request.json['img'])
    names, faces, landmarks = recognition_handle(img)
    return make_response(jsonify({'Status: ': 'finished', 'names': json.dumps(names), 'faces': json.dumps(faces.tolist()), 'landmarks': json.dumps(landmarks.tolist())}), 200)

@app.route('/model/api/v1.0/add_face', methods=['POST'])
def add_face():
    if not request.json or not 'img' in request.json or not 'label' in request.json:
       abort(204)
    img = decode_img(request.json['img'])
    label = request.json['label']
    dataset_add(img, label)
    return make_response(jsonify({'Status: ': 'finished'}), 200)

if __name__ == '__main__':
    app.run(host='0.0.0.0')

