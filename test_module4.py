import cv2
import matplotlib.pyplot as plt
from module4.feature_extraction import (
    extract_hog_features,
    extract_hu_moments,
    extract_histogram_features
)

# Load test image
img = cv2.imread('test_images/test1.jpeg')
if img is None:
    raise FileNotFoundError("⚠️ Image not found! Check your 'test_images/digit_sample.jpg' path.")

# 1️⃣ HOG Features
features, hog_img = extract_hog_features(img, visualize=True)
print(f"HOG feature vector length: {len(features)}")

plt.imshow(hog_img, cmap='gray')
plt.title("HOG Visualization")
plt.axis('off')
plt.show()

# 2️⃣ Hu Moments
hu = extract_hu_moments(img)
print("Hu Moments:", hu)

# 3️⃣ Color Histogram Features
hist_features = extract_histogram_features(img)
print(f"Histogram feature vector length: {len(hist_features)}")
