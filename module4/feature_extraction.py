import cv2
import numpy as np
from skimage.feature import hog
from skimage.measure import moments_hu


# ---------- Feature Extraction ----------
def extract_hog_features(img, visualize=False):
    """
    Extract Histogram of Oriented Gradients (HOG) features.
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    features, hog_image = hog(
        img_gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=True
    )
    if visualize:
        return features, np.uint8(hog_image * 255)
    return features


def extract_hu_moments(img):
    """
    Extract 7 Hu moments — shape descriptors invariant to scale and rotation.
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    moments = cv2.moments(img_gray)
    huMoments = cv2.HuMoments(moments).flatten()
    return huMoments


def extract_histogram_features(img):
    """
    Extract color histogram features from each channel.
    """
    chans = cv2.split(img)
    features = []
    for chan in chans:
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        features.extend(hist)
    return np.array(features)
