import cv2
from module2.segmentation import (
    global_threshold,
    adaptive_threshold,
    otsu_threshold,
    sobel_edge,
    canny_edge,
    morphological_ops,
    find_contours,
    show_comparison
)

# Load test image
img = cv2.imread('test_images/test1.jpeg')
if img is None:
    raise FileNotFoundError("⚠️ Image not found! Check 'test_images/digit_sample.jpg'.")

# 1️⃣ Global Thresholding
global_bin = global_threshold(img)
show_comparison(img, global_bin, "Original", "Global Threshold")

# 2️⃣ Adaptive Thresholding
adaptive_bin = adaptive_threshold(img)
show_comparison(img, adaptive_bin, "Original", "Adaptive Threshold")

# 3️⃣ Otsu Thresholding
otsu_bin = otsu_threshold(img)
show_comparison(img, otsu_bin, "Original", "Otsu Threshold")

# 4️⃣ Sobel Edge
sobel_img = sobel_edge(img)
show_comparison(img, sobel_img, "Original", "Sobel Edge")

# 5️⃣ Canny Edge
canny_img = canny_edge(img)
show_comparison(img, canny_img, "Original", "Canny Edge")

# 6️⃣ Morphological Operations (Dilation)
dilated = morphological_ops(otsu_bin, 'dilate', kernel_size=3)
show_comparison(otsu_bin, dilated, "Original (Otsu)", "Dilated")

# 7️⃣ Contour Detection
contours = find_contours(img)
show_comparison(img, contours, "Original", "Contours")
