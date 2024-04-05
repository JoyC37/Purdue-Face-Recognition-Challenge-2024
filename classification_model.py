import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
import os
from PIL import Image
import cv2

#Defining necessary variables
input_path = '/content/data/categorized_data/categorized_data' #Google colab was used and the images extracted from data_preprocessing were uploaded to drive
test_path = '/content/data/test_data/test_data' #The preprocessed test data was also uploaded to drive
batch_size = 32
image_shape = (224,224,3)
epochs = 50

#Initialising generator for batches of training and validation data
#This code is based on https://www.tensorflow.org/tutorials/images/classification
train_ds = tf.keras.utils.image_dataset_from_directory(input_path,
                                                      image_size=image_shape[:-1],
                                                      seed=42,
                                                      subset='training',
                                                      validation_split=0.2,
                                                      batch_size=batch_size,
                                                      label_mode='categorical')

val_ds = tf.keras.utils.image_dataset_from_directory(input_path,
                                                      image_size=image_shape[:-1],
                                                      seed=42,
                                                      subset='validation',
                                                      validation_split=0.2,
                                                      batch_size=batch_size,
                                                      label_mode='categorical')


#Storing the class names
class_names = train_ds.class_names
num_classes = len(class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

#The pre-trained ResNet101 model was used and fine-tuned to suit the current dataset
base_model = tf.keras.applications.ResNet101(include_top=False, weights='imagenet', classes=num_classes, input_shape=image_shape)

#To apply image augmentation with the aim to reduce overfitting
augmentation = tf.keras.Sequential([
    layers.RandomFlip('horizontal',input_shape = image_shape),
    layers.RandomFlip('vertical'),
    layers.RandomZoom(0.1)
])

#Defines the layers that should be fine-tuned
base_model.trainable = True
limit = 100
for layer in base_model.layers[:limit]:
  layer.trainable = False

#Creating the actual model
model = tf.keras.models.Sequential([
    augmentation,
    base_model,

    layers.Dropout(0.5),
    layers.GlobalAveragePooling2D(),

    layers.Dense(512, activation='relu'),

    layers.Dense(num_classes, activation='softmax')
])

#Model is generated with categorical_crossentropy as labels are in categorical mode
model.compile(optimizer= tf.keras.optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_ds, validation_data=val_ds, epochs=15) #50 epochs was not used as the model overfits

#Visualizing training and validation metrics
plt.plot(history.history['accuracy'], label='training')
plt.plot(history.history['val_accuracy'], label='validation')
plt.title("Model accuracy")
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='training')
plt.plot(history.history['val_loss'], label='validation')
plt.title("Model loss")
plt.legend()
plt.show()

#Generator for test data batches; labels are set to None as ground truth labels are unavailable; shuffle is set to False to avoid shuffling of filename order during prediction
test_ds = tf.keras.utils.image_dataset_from_directory(test_path,
                                                      image_size=image_shape[:-1],
                                                      labels=None,
                                                      shuffle=False,
                                                      batch_size=batch_size)

fnames = test_ds.file_paths #stores the entire file path as filename
fnames = [name[34:-4] for name in fnames] #slices the string to only store the image number
fnames[:6]

#Model prediction - the output provides the probabilities for each of the 100 classes - must be converted back to class name
newpred = model.predict(test_ds)

#To convert back to class names - we choose the class index with the highest probability
new_indices = np.argmax(newpred,axis=1)
new_indices.shape

#The class names are generated based on the new_indices list
celeb_names = []
for index in new_indices:
  celeb_names.append(class_names[index])

#Creating the submission file in required format
pred_file = np.hstack([np.array(fnames).reshape((len(fnames),1)),np.array(celeb_names).reshape((len(celeb_names),1))])
pred_file = pred_file.astype(str)
pred_file = np.vstack([['Id','Category'],pred_file])
pred_file[:6]
