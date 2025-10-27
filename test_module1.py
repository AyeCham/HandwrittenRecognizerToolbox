import cv2
from module1.enhancement import (
    histogram_equalization,
    adjust_brightness_contrast,
    mean_filter,
    median_filter,
    laplacian_sharpen,
    frequency_filter,
    show_comparison
)

# Load test image
img = cv2.imread('test_images/test1.jpeg')
if img is None:
    raise FileNotFoundError("⚠️ Image not found! Check your 'test_images/digit_sample.jpg' path.")

# 1️⃣ Histogram Equalization
eq_img = histogram_equalization(img)
show_comparison(img, eq_img, "Original", "Histogram Equalization")

# 2️⃣ Brightness & Contrast Adjustment
bc_img = adjust_brightness_contrast(img, brightness=30, contrast=50)
show_comparison(img, bc_img, "Original", "Brightness & Contrast")

# 3️⃣ Mean Filter
mean_img = mean_filter(img, 5)
show_comparison(img, mean_img, "Original", "Mean Filter")

# 4️⃣ Median Filter
median_img = median_filter(img, 5)
show_comparison(img, median_img, "Original", "Median Filter")

# 5️⃣ Laplacian Sharpening
sharp_img = laplacian_sharpen(img)
show_comparison(img, sharp_img, "Original", "Laplacian Sharpen")

# 6️⃣ Frequency Low-Pass Filter
freq_low = frequency_filter(img, 'low', 30)
show_comparison(img, freq_low, "Original", "Low-Pass Filter")

# 7️⃣ Frequency High-Pass Filter
freq_high = frequency_filter(img, 'high', 30)
show_comparison(img, freq_high, "Original", "High-Pass Filter")
