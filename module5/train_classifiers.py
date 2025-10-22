# train_classifiers.py
import os
import joblib
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from module4_features.features import extract_hog, preprocess_for_recognition

def load_digits_features(use_hog=True):
    # use sklearn digits as demo (8x8) or use torchvision MNIST later (28x28)
    data = load_digits()
    X = data.images
    y = data.target
    feats = []
    for img in X:
        # scale up to 28x28 for HOG compatibility or use as-is
        img28 = cv2_resize_to_28(img)  # helper below
        if use_hog:
            f = extract_hog(img28, pixels_per_cell=(4,4))
        else:
            f = img28.flatten()
        feats.append(f)
    return np.array(feats), y

def cv2_resize_to_28(img):
    import cv2
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    img8 = np.uint8(img * 255)
    return cv2.resize(img8, (28,28), interpolation=cv2.INTER_AREA)

def train_and_save(out_dir='models'):
    os.makedirs(out_dir, exist_ok=True)
    X, y = load_digits_features(use_hog=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # KNN
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    ypred = knn.predict(X_test)
    print("KNN acc:", accuracy_score(y_test, ypred))
    joblib.dump(knn, os.path.join(out_dir, 'knn_model.joblib'))
    # ANN (MLP)
    mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=400, random_state=42)
    mlp.fit(X_train, y_train)
    ypred2 = mlp.predict(X_test)
    print("ANN acc:", accuracy_score(y_test, ypred2))
    joblib.dump(mlp, os.path.join(out_dir, 'mlp_model.joblib'))
    print("Saved models to", out_dir)
    print(classification_report(y_test, ypred2))
    print(confusion_matrix(y_test, ypred2))

if __name__ == '__main__':
    train_and_save()
