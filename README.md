# Purdue-Face-Recognition-Challenge-2024

## For the teaching staff of ECE50024 (Purdue University)

The training and test data were individually processed using the data_preprocessing file. For the training set, the images were stored in subfolders having the same name as the corresponding celebrity name. For the test set, only the face extraction using "haarcascade" was performed.

The processed datasets were uploaded to Google Drive and Kaggle (as a private dataset) to utilize their GPU-enable notebooks.

The classification_model.py file provides the code for the final model that provided me with the best accuracy. Previously, I had tested various CNN architectures, VGG16, and ResNet50 models (with RGB and grayscale images of various dimensions).
