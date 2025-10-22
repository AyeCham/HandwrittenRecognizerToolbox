# segmentation.py
import cv2
import numpy as np
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu

def sobel_edge(img):
    gray = img if len(img.shape)==2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    g = np.sqrt(gx*gx + gy*gy)
    g = np.uint8(255 * (g / (np.max(g) + 1e-8)))
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_OTSU)
    return bw

def prewitt_edge(img):
    gray = img if len(img.shape)==2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernelx = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])
    kernely = kernelx.T
    gx = cv2.filter2D(gray, -1, kernelx)
    gy = cv2.filter2D(gray, -1, kernely)
    g = np.sqrt(gx.astype(float)**2 + gy.astype(float)**2)
    g = np.uint8(255 * (g / (np.max(g) + 1e-8)))
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_OTSU)
    return bw

def canny_edge(img, low=100, high=200):
    gray = img if len(img.shape)==2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(gray, low, high)

def otsu_threshold(img):
    gray = img if len(img.shape)==2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th = threshold_otsu(gray)
    bw = (gray > th).astype(np.uint8) * 255
    return bw

def region_growing(img, seed, tol=0.05):
    """Simple region growing on normalized grayscale image (0..1)."""
    gray = img.astype(np.float32)
    if gray.max() > 1:
        gray = gray / 255.0
    h,w = gray.shape[:2]
    output = np.zeros((h,w), dtype=np.uint8)
    visited = np.zeros((h,w), dtype=bool)
    stack = [seed]
    seed_val = gray[seed[1], seed[0]]
    while stack:
        x,y = stack.pop()
        if x<0 or y<0 or x>=w or y>=h: 
            continue
        if visited[y,x]:
            continue
        visited[y,x] = True
        if abs(gray[y,x] - seed_val) <= tol:
            output[y,x] = 255
            for dx,dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                stack.append((x+dx,y+dy))
    return output

def color_kmeans_segmentation(img, k=3, colorspace='rgb'):
    imgf = img.astype(np.float32) / 255.0
    if colorspace.lower() == 'hsv':
        imgf = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32) / 255.0
    Z = imgf.reshape((-1,3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(Z, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    labels = labels.flatten()
    segmented = centers[labels].reshape(img.shape)
    segmented = np.uint8(segmented * 255)
    return segmented

def morphological_ops(binary_img, op='open', kernel_size=3):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    if op == 'dilate':
        return cv2.dilate(binary_img, kernel, iterations=1)
    elif op == 'erode':
        return cv2.erode(binary_img, kernel, iterations=1)
    elif op == 'open':
        return cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)
    elif op == 'close':
        return cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)
    else:
        raise ValueError("op must be one of 'dilate','erode','open','close'")
