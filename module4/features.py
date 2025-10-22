# features.py
import numpy as np
import cv2
from skimage.feature import hog

def preprocess_for_recognition(img, size=(28,28), deskew=False):
    # convert to grayscale and resize
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img.copy()
    # optionally binarize / invert if necessary
    img_resized = cv2.resize(img_gray, size, interpolation=cv2.INTER_AREA)
    return img_resized

def extract_hog(img, pixels_per_cell=(4,4)):
    # img should be grayscale 28x28
    img = img.astype('float32') / 255.0
    feat = hog(img, pixels_per_cell=pixels_per_cell, cells_per_block=(2,2), orientations=9, feature_vector=True)
    return feat

def flatten_image(img):
    img = img.astype('float32') / 255.0
    return img.flatten()
