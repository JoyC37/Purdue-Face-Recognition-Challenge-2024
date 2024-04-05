import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from PIL import Image

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
input_path = 'train/train' #test data was processed using similar approach
output_path = 'categorized_data'
training_labels = pd.read_csv('train/train.csv')
training_labels.head()

for filename in os.listdir(path):
    if filename.endswith(('.jpg','.jpeg','.png')):
        image = Image.open(os.path.join(path,filename)) #PIL used as cv2 lead to some file read errors
        if image.mode != 'RGB':
            image = image.convert('RGB')
        img = np.array(image)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        label = training_labels.loc[training_labels['File Name']==filename,'Category'].values[0] #to create individual folders for celebrities
        
        if not os.path.exists(os.path.join(output_path,label)):
            os.mkdir(os.path.join(output_path,label)) #creates folder if it does not exist

        #This code is based on https://towardsdatascience.com/face-detection-in-2-minutes-using-opencv-python-90f89d7c0f81 and https://stackoverflow.com/questions/52288066/how-to-limit-number-of-faces-detected-by-haar-cascades
        if img is not None:
            detected_faces = face_cascade.detectMultiScale(gray,1.2,10,minSize=(64,64),flags=cv2.CASCADE_SCALE_IMAGE)
            if detected_faces is None:
                img = cv2.resize(img, (128,128))
                cv2.imwrite(os.path.join(output_path,label,filename),img)
            else:
                largest_face = img #stores only the largest face detected
                max_area = 0
                for (x,y,w,h) in detected_faces:
                    face_area = w*h
                    if face_area > max_area:
                        largest_face = img[y:y+h,x:x+w]
                        max_area = face_area
                        
                cv2.imwrite(os.path.join(output_path,label,filename),largest_face)
        else:
            print("Unable to access file {}".format(filename))
