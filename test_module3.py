import cv2
from module3.transform import (
    translate,
    rotate,
    scale,
    affine_transform,
    perspective_transform,
    resize_with_interpolation,
    show_comparison
)

# Load test image
img = cv2.imread('test_images/test1.jpeg')
if img is None:
    raise FileNotFoundError("⚠️ Image not found! Check 'test_images/digit_sample.jpg'.")

# 1️⃣ Translation
translated = translate(img, tx=50, ty=30)
show_comparison(img, translated, "Original", "Translated (+50, +30)")

# 2️⃣ Rotation
rotated = rotate(img, angle=30)
show_comparison(img, rotated, "Original", "Rotated 30°")

# 3️⃣ Scaling
scaled = scale(img, fx=1.5, fy=1.5)
show_comparison(img, scaled, "Original", "Scaled ×1.5")

# 4️⃣ Affine Transformation
affined = affine_transform(img)
show_comparison(img, affined, "Original", "Affine Transform")

# 5️⃣ Perspective Transformation
persp = perspective_transform(img)
show_comparison(img, persp, "Original", "Perspective Transform")

# 6️⃣ Interpolation Resizing (Bicubic)
resized = resize_with_interpolation(img, scale_percent=200, method='bicubic')
show_comparison(img, resized, "Original", "Bicubic Resized ×2")
