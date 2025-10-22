# classifier.py
import os
import joblib
import numpy as np
from module4_features.features import preprocess_for_recognition, extract_hog, flatten_image

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
KNN_PATH = os.path.join(MODELS_DIR, 'knn_model.joblib')
MLP_PATH = os.path.join(MODELS_DIR, 'mlp_model.joblib')

def load_models():
    knn = joblib.load(KNN_PATH)
    mlp = joblib.load(MLP_PATH)
    return knn, mlp

def predict(img, model='mlp', use_hog=True):
    # img: BGR or grayscale image
    img_p = preprocess_for_recognition(img, size=(28,28))
    feat = extract_hog(img_p) if use_hog else flatten_image(img_p)
    feat = feat.reshape(1, -1)
    knn, mlp = load_models()
    if model == 'knn':
        return knn.predict(feat)[0]
    else:
        return mlp.predict(feat)[0]
