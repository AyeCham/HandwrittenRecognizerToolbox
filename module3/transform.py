import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---------- Basic Transformations ----------

def translate(img, tx, ty):
    """Shift image by tx, ty pixels."""
    rows, cols = img.shape[:2]
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    shifted = cv2.warpAffine(img, M, (cols, rows))
    return shifted


def rotate(img, angle, scale=1.0):
    """Rotate image by given angle around its center."""
    rows, cols = img.shape[:2]
    center = (cols // 2, rows // 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(img, M, (cols, rows))
    return rotated


def scale(img, fx, fy, interpolation=cv2.INTER_LINEAR):
    """Resize image by fx, fy scaling factors."""
    return cv2.resize(img, None, fx=fx, fy=fy, interpolation=interpolation)


# ---------- Affine & Perspective Transformations ----------

def affine_transform(img, pts1=None, pts2=None):
    """Apply affine transformation using given point sets."""
    rows, cols = img.shape[:2]
    if pts1 is None:
        pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
    if pts2 is None:
        pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
    M = cv2.getAffineTransform(pts1, pts2)
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst


def perspective_transform(img, pts1=None, pts2=None):
    """Apply perspective (homography) transformation."""
    rows, cols = img.shape[:2]
    if pts1 is None:
        pts1 = np.float32([[0, 0], [cols-1, 0], [0, rows-1], [cols-1, rows-1]])
    if pts2 is None:
        pts2 = np.float32([[50, 50], [cols-100, 30], [30, rows-80], [cols-50, rows-50]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, (cols, rows))
    return dst


# ---------- Interpolation Techniques ----------

def resize_with_interpolation(img, scale_percent=150, method='bilinear'):
    """Resize image with chosen interpolation."""
    methods = {
        'nearest': cv2.INTER_NEAREST,
        'bilinear': cv2.INTER_LINEAR,
        'bicubic': cv2.INTER_CUBIC,
    }
    interp = methods.get(method, cv2.INTER_LINEAR)
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    resized = cv2.resize(img, (width, height), interpolation=interp)
    return resized


# ---------- Visualization ----------
def show_comparison(original, processed, title1="Original", title2="Transformed"):
    """Display two images side-by-side."""
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    if len(original.shape) == 3:
        plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(original, cmap='gray')
    plt.title(title1)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    if len(processed.shape) == 3:
        plt.imshow(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(processed, cmap='gray')
    plt.title(title2)
    plt.axis('off')
    plt.show()
