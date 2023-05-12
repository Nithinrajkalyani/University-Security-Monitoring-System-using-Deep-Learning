# Importing the libraries
from PIL import Image
from keras.applications.vgg16 import preprocess_input
import base64
from io import BytesIO
import json
import random
import cv2
from keras.models import load_model
import numpy as np
import time
import os 
import datetime

current_time = datetime.datetime.now().strftime("%d-%m-%y_%H-%M-%S")

# Create directories to store known and unknown person images
known_dir = 'C:/Mini/Attendance/known_persons'
unknown_dir = 'C:/Mini/Attendance/unknown_persons'
os.makedirs(known_dir, exist_ok=True)
os.makedirs(unknown_dir, exist_ok=True)

# Load the pre-trained model
model = load_model('Reva_Vgg_Model_V1.h5')

# Loading the cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

names = {'0': 'Kamal', '1': 'Shahnawaz', '2': 'Rehan', '3': 'Manisha', '4': 'NithinRaj', '5': 'Ashfaq', '6': 'Prudivi', '7': 'Suraj'}

# Define the face detection function
def detect_face(frame):
    # Function detects faces and returns the cropped face
    # If no face detected, it returns None

    faces = face_cascade.detectMultiScale(frame, 1.3, 5)
    if faces is ():
        return None, None

    # Crop all faces found
    for (x,y,w,h) in faces:
        #adding rectangle around your face
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        cropped_face = frame[y:y+h, x:x+w]
        # Resize the image to the size required by the model
        resized_face = cv2.resize(cropped_face, (224, 224))
        # Convert the image to a NumPy array
        img_array = np.array(resized_face, dtype=np.float32)
        # Normalize the image
        img_array /= 255.0
        # Expand the dimensions to match the input shape of the model
        img_array = np.expand_dims(img_array, axis=0)
        # Perform the prediction using the model
        pred = model.predict(img_array)
        # Extract the probability of detection
        probability = np.max(pred)
        # Set threshold for probability
        threshold = 0.9
        if probability > threshold:
            idx = np.argmax(pred)
            name = names[str(idx)]
            cv2.putText(frame,name, (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            cv2.imwrite(os.path.join(known_dir, f'{name}_{current_time}.jpg'), cropped_face)
        else:
            name="Unknown person"
            cv2.putText(frame,name, (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            cv2.imwrite(os.path.join(unknown_dir, f'unknown_{current_time}.jpg'), cropped_face)
            
    return frame, probability

# Initialize the camera
cap = cv2.VideoCapture(0)

# Set the delay in seconds
delay = 3

while True:
    # Capture a frame from the camera
    ret, frame = cap.read()
    if not ret:
        break
    # Perform face detection on the frame
    detected_frame, probability = detect_face(frame)
    if detected_frame is None:
        continue
    # Display the frame
    cv2.imshow('Video', detected_frame)
    # Check for 'q' key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # Delay for a specified number of seconds
    time.sleep(delay)

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()