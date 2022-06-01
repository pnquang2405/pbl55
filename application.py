
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from asyncore import loop
from cv2 import VideoCapture

import tensorflow as tf
from imutils.video import VideoStream

from datetime import datetime
from re import T
import argparse
import facenet
import imutils
import os
import sys
import math
import pickle
import align.detect_face
import numpy as np
import cv2
import collections
from sklearn.svm import SVC
import pyrebase
import time
# import cv2 
from flask import Flask, render_template, Response




firebaseConfig = {
    'apiKey': "AIzaSyAwO4hBvYnKCPp5_0tWw4BeK1eEgH71IeY",
    'authDomain': "mypbl5-88973.firebaseapp.com",
    'databaseURL': "https://mypbl5-88973-default-rtdb.firebaseio.com",
    'projectId': "mypbl5-88973",
    'storageBucket': "mypbl5-88973.appspot.com",
    'messagingSenderId': "233060726166",
    'appId': "1:233060726166:web:94afbe5b70a70c534c99b7",
    'measurementId': "G-ZC4HB3ZT63"   
    }
firebase = pyrebase.initialize_app(firebaseConfig)
database=firebase.database()
storage = firebase.storage()
    

# with open('labels', 'rb') as f:
# 	dict = pickle.load(f)
# 	f.close()

# cam = cv2.VideoCapture(0)
# cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


def getNumberUser():
        count=0
        all_users = database.child("User").get()
        if all_users==None:
                return count
        for users in all_users.each():
                count+=1
        return count

font = cv2.FONT_HERSHEY_SIMPLEX
count1=getNumberUser()+1
print(count1)

def SentData(name,frame,count1):
        name1=name+str(count1)+".jpg"
        now=datetime.now()
        dt=now.strftime("%d/%m/%Y :%H:%M")
        print(dt)
        cv2.imwrite(name1,frame)
        auth = firebase.auth()
        email = "thanhthan2k1@gmail.com"
        password = "thanhthan"
        user = auth.sign_in_with_email_and_password(email, password)
        storage.child(name1).put(name1)
                #database=firebase.database()
                
                #user = database.child("Users");
                
                #database.child("User"+str(count))
                
        # Enter your user account details 
        

        imageUrl = storage.child(name1).get_url(user['idToken'])
                #print(imageUrl)
        data={"avatar":imageUrl,"name":name,"time":dt}
        #database.set(data)
        database.child("User").child("User"+str(count1))
                #user.setValue(data);
        database.set(data)
        print("da gui anh")
        os.remove(name1)
        count=count+1
                #sleep(1)

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--path', help='Path of the video you want to test on.', default=0)
#     args = parser.parse_args()

#     MINSIZE = 20
#     THRESHOLD = [0.6, 0.7, 0.7]
#     FACTOR = 0.709
#     IMAGE_SIZE = 182
#     INPUT_IMAGE_SIZE = 160
#     CLASSIFIER_PATH = 'models/facemodel.pkl' 
#     VIDEO_PATH = args.path
#     FACENET_MODEL_PATH = 'models/20180402-114759.pb'

#     # Load The Custom Classifier
#     with open(CLASSIFIER_PATH, 'rb') as file:
#         model, class_names = pickle.load(file)
#     print("Custom Classifier, Successfully loaded")

#     with tf.Graph().as_default():

#         gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
#         sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))

#         with sess.as_default():

#             # Load the model
#             print('Loading feature extraction model')
#             facenet.load_model(FACENET_MODEL_PATH)  

#             # Get input and output tensors
#             images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
#             embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
#             phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
#             embedding_size = embeddings.get_shape()[1]

#             pnet, rnet, onet = align.detect_face.create_mtcnn(sess, "src/align")

#             people_detected = set()
#             person_detected = collections.Counter()

#             cap  = VideoStream(1).start()

#             check = False
#             name = ""
#             while (True): 
#                 frame = cap.read()
#                 frame = imutils.resize(frame, width=600)
#                 frame = cv2.flip(frame, 1)

#                 bounding_boxes, _ = align.detect_face.detect_face(frame, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)

#                 faces_found = bounding_boxes.shape[0]
#                 try:
#                     if faces_found > 1:
#                         cv2.putText(frame, "Only one face", (0, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL,
#                                     1, (255, 255, 255), thickness=1, lineType=2)
#                     elif faces_found > 0:
#                         det = bounding_boxes[:, 0:4]
#                         bb = np.zeros((faces_found, 4), dtype=np.int32)
#                         for i in range(faces_found):
#                             bb[i][0] = det[i][0]
#                             bb[i][1] = det[i][1]
#                             bb[i][2] = det[i][2]
#                             bb[i][3] = det[i][3]
#                             print(bb[i][3]-bb[i][1])
#                             print(frame.shape[0])
#                             print((bb[i][3]-bb[i][1])/frame.shape[0])
#                             if (bb[i][3]-bb[i][1])/frame.shape[0]>0.25: 
#                                 cropped = frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :]
#                                 scaled = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE),
#                                                     interpolation=cv2.INTER_CUBIC)
#                                 scaled = facenet.prewhiten(scaled)
#                                 scaled_reshape = scaled.reshape(-1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3)
#                                 feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
#                                 emb_array = sess.run(embeddings, feed_dict=feed_dict)

#                                 predictions = model.predict_proba(emb_array)
#                                 best_class_indices = np.argmax(predictions, axis=1)
#                                 best_class_probabilities = predictions[
#                                     np.arange(len(best_class_indices)), best_class_indices]
#                                 best_name = class_names[best_class_indices[0]]
#                                 print("Name: {}, Probability: {}".format(best_name, best_class_probabilities))

#                                 if best_class_probabilities > 0.8:
#                                     cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)
#                                     text_x = bb[i][0]
#                                     text_y = bb[i][3] + 20

#                                     name = class_names[best_class_indices[0]]
#                                     cv2.putText(frame, name, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
#                                                 1, (255, 255, 255), thickness=1, lineType=2)
#                                     cv2.putText(frame, str(round(best_class_probabilities[0], 3)), (text_x, text_y + 17),
#                                                 cv2.FONT_HERSHEY_COMPLEX_SMALL,
#                                                 1, (255, 255, 255), thickness=1, lineType=2)
#                                     person_detected[best_name] += 1
#                                     print(name)
#                                     print("da vao day")
#                                     check = True
#                                     break
#                                 else:
#                                     if check == False:
#                                         name = "Unknown"
#                                     # SentData(name,frame,count1)

#                 except:
#                     pass

#                 cv2.imshow('Face Recognition', frame)
#                 if cv2.waitKey(1) & 0xFF == ord('q'):
#                     break

#             # SentData(name,frame,count1)
#             cap.stop()
#             cv2.destroyWindow()

application = Flask(__name__)


@application.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


def gen():
    # """Video streaming generator function."""
    # cap = cv2.VideoCapture('768x576.avi')

    # # Read until video is completed
    # while(cap.isOpened()):
    #   # Capture frame-by-frame
    #     ret, img = cap.read()
    #     if ret == True:
    #         img = cv2.resize(img, (0,0), fx=0.5, fy=0.5) 
    #         frame = cv2.imencode('.jpg', img)[1].tobytes()
    #         yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    #         time.sleep(0.1)
    #     else: 
    #         break
        
    

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--path', help='Path of the video you want to test on.', default=0)
    # args = parser.parse_args()

    MINSIZE = 20
    THRESHOLD = [0.6, 0.7, 0.7]
    FACTOR = 0.709
    # IMAGE_SIZE = 182
    INPUT_IMAGE_SIZE = 160
    CLASSIFIER_PATH = 'facemodel.pkl' 
    # VIDEO_PATH = args.path
    FACENET_MODEL_PATH = '20180402-114759.pb'

    # Load The Custom Classifier
    with open(CLASSIFIER_PATH, 'rb') as file:
        model, class_names = pickle.load(file)
    print("Custom Classifier, Successfully loaded")

    with tf.Graph().as_default():

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))

        with sess.as_default():

            # Load the model
            print('Loading feature extraction model')
            facenet.load_model(FACENET_MODEL_PATH)  

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, "align")

            people_detected = set()
            person_detected = collections.Counter()

            cap  = VideoStream('https://filesamples.com/samples/video/mjpeg/sample_1280x720_surfing_with_audio.mjpeg').start()

            check = False
            name = ""
            while (True): 
                frame = cap.read()
                # frame = imutils.resize(frame, width=600)
                frame = cv2.flip(frame, 1)

                bounding_boxes, _ = align.detect_face.detect_face(frame, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)

                faces_found = bounding_boxes.shape[0]
                try:
                    if faces_found > 1:
                        cv2.putText(frame, "Only one face", (0, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                    1, (255, 255, 255), thickness=1, lineType=2)
                    elif faces_found > 0:
                        det = bounding_boxes[:, 0:4]
                        bb = np.zeros((faces_found, 4), dtype=np.int32)
                        for i in range(faces_found):
                            bb[i][0] = det[i][0]
                            bb[i][1] = det[i][1]
                            bb[i][2] = det[i][2]
                            bb[i][3] = det[i][3]
                            print(bb[i][3]-bb[i][1])
                            print(frame.shape[0])
                            print((bb[i][3]-bb[i][1])/frame.shape[0])
                            if (bb[i][3]-bb[i][1])/frame.shape[0]>0.25: 
                                cropped = frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :]
                                scaled = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE),
                                                    interpolation=cv2.INTER_CUBIC)
                                scaled = facenet.prewhiten(scaled)
                                scaled_reshape = scaled.reshape(-1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3)
                                feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
                                emb_array = sess.run(embeddings, feed_dict=feed_dict)

                                predictions = model.predict_proba(emb_array)
                                best_class_indices = np.argmax(predictions, axis=1)
                                best_class_probabilities = predictions[
                                    np.arange(len(best_class_indices)), best_class_indices]
                                best_name = class_names[best_class_indices[0]]
                                print("Name: {}, Probability: {}".format(best_name, best_class_probabilities))

                                if best_class_probabilities > 0.8:
                                    cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)
                                    text_x = bb[i][0]
                                    text_y = bb[i][3] + 20

                                    name = class_names[best_class_indices[0]]
                                    cv2.putText(frame, name, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                1, (255, 255, 255), thickness=1, lineType=2)
                                    cv2.putText(frame, str(round(best_class_probabilities[0], 3)), (text_x, text_y + 17),
                                                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                1, (255, 255, 255), thickness=1, lineType=2)
                                    person_detected[best_name] += 1
                                    print(name)
                                    print("da vao day")
                                    check = True
                                    break
                                else:
                                    if check == False:
                                        name = "Unknown"
                                    # SentData(name,frame,count1)

                except:
                    pass

                # cv2.imshow('Face Recognition', frame)
                frame = cv2.imencode('.jpg', frame)[1].tobytes()
                yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                #time.sleep(0.1)
                key = cv2.waitKey(20)
                if key == 27:
                    break

            # SentData(name,frame,count1)
            # cap.stop()
            # cv2.destroyWindow()


@application.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
# application.debug = True
# application.run(port=5004)    