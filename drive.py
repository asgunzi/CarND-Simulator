import argparse
import base64
import json
import cv2 

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import time
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO

from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array

# Fix error with Keras and TensorFlow
import tensorflow as tf
tf.python.control_flow_ops = tf


sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

size1 = 64
size2 =96

prop1 = 40/160
propx1 = 10/320
propx2 = 310/320


@sio.on('telemetry')
def telemetry(sid, data):
    # The current steering angle of the car
    steering_angle = data["steering_angle"]
    # The current throttle of the car
    throttle = data["throttle"]
    # The current speed of the car
    speed = data["speed"]
    # The current image from the center camera of the car
    imgString = data["image"]
    image = Image.open(BytesIO(base64.b64decode(imgString)))
    image_a = np.asarray(image)

    ##Grayscale
    ##image_array =  cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY) 
    
    # #HSV
    # image_hsv =  cv2.cvtColor(image_a, cv2.COLOR_RGB2HSV) 
    # image_hsv =  cv2.cvtColor(image_a, cv2.COLOR_RGB2YUV) 

    # image_array =image_hsv[:,:,1].squeeze()


    #Full color
    image_array =image_a[:,:,:].squeeze()

    #Edge
    #low_threshold =100
    #high_threshold =200
    #edge_img = cv2.Canny(image_array, low_threshold, high_threshold)

    crp1_y = int(prop1*image_array.shape[0])
    crp2_y = int(image_array.shape[0])
    
    crp1_x = int(propx1*image_array.shape[1])
    crp2_x = int(propx2*image_array.shape[1])

    crop_img = image_array[crp1_y:crp2_y,crp1_x:crp2_x,:] 
    # crop_img = image_array[crp1_y:crp2_y,crp1_x:crp2_x] 

    # image_array2= np.zeros((size1,size2,1))
    # image_array2[:,:,0] = cv2.resize(crop_img, (size2,size1))
    image_array2= np.zeros((size1,size2,3))
    image_array2[:,:,:] = cv2.resize(crop_img, (size2,size1))

    transformed_image_array = image_array2[None, :, :, :]
	

	
    # This model currently assumes that the features of the model are just the images. Feel free to change this.
    with tf.device("/cpu:0"):
        steering_angle = float(model.predict(transformed_image_array, batch_size=1))
    # The driving model currently just outputs a constant throttle. Feel free to edit this.
    # if abs(steering_angle) < .1:
    #     throttle = 0.25
    # else:
        throttle = 0.2       

    print(steering_angle, throttle)
    send_control(steering_angle, throttle)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit("steer", data={
    'steering_angle': steering_angle.__str__(),
    'throttle': throttle.__str__()
    }, skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
    help='Path to model definition json. Model weights should be on the same path.')
    args = parser.parse_args()
    with open(args.model, 'r') as jfile:
        # NOTE: if you saved the file by calling json.dump(model.to_json(), ...)
        # then you will have to call:
        #
        #   model = model_from_json(json.loads(jfile.read()))\
        #
        # instead.
        model = model_from_json(jfile.read())


    model.compile("adam", "mse")
    weights_file = args.model.replace('json', 'h5')
    model.load_weights(weights_file)

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
